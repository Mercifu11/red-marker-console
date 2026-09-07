#!/usr/bin/env python3
"""Scan a video for red pixels inside a rectangular region.

Every N-th frame is decoded, only the configured region is kept, and each kept
frame is tested for pixels whose red channel is above a threshold and whose HSV
hue is close to red. Consecutive positive samples are merged into one event.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

SnapshotCallback = Callable[[float, bytes], None]


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    codec: str


@dataclass(frozen=True)
class Event:
    start_time: float
    duration: float
    red_pixels: int


class ScanError(Exception):
    pass


def _find_binary(env_name: str, name: str) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    found = shutil.which(name)
    return found or name


FFMPEG = _find_binary("FFMPEG_BIN", "ffmpeg")
FFPROBE = _find_binary("FFPROBE_BIN", "ffprobe")


def parse_region(text: str) -> Region:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "region must be x,y,width,height, for example 716,772,387,111"
        )
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "region values must be integers"
        ) from exc
    region = Region(*values)
    if region.width <= 0 or region.height <= 0:
        raise argparse.ArgumentTypeError("region width and height must be positive")
    return region


def _parse_rational(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            if int(denominator) == 0:
                return None
            return int(numerator) / int(denominator)
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_positive(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None and value > 0:
            return value
    return None


def probe_video(path: str) -> VideoInfo:
    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,nb_frames,duration",
        "-show_format",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise ScanError(
            "ffprobe failed: " + (result.stderr or "").strip()
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ScanError("could not parse ffprobe output") from exc

    streams = data.get("streams") or []
    if not streams:
        raise ScanError("no video stream found in " + path)
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScanError("video dimensions could not be read") from exc

    fps = (
        _parse_rational(stream.get("avg_frame_rate"))
        or _parse_rational(stream.get("r_frame_rate"))
    )
    if fps is None or fps <= 0:
        raise ScanError("video frame rate could not be read")

    nb_frames = _first_positive(_to_float(stream.get("nb_frames")))
    format_data = data.get("format") or {}
    duration = _first_positive(
        _to_float(stream.get("duration")),
        _to_float(format_data.get("duration")),
        nb_frames / fps if nb_frames else None,
    )
    if duration is None:
        duration = 0.0

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        codec=str(stream.get("codec_name") or "unknown"),
    )


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def validate_region(info: VideoInfo, region: Region) -> None:
    if region.x < 0 or region.y < 0:
        raise ScanError("region x and y must be non-negative")
    if region.x + region.width > info.width:
        raise ScanError(
            f"region right edge {region.x + region.width} exceeds width {info.width}"
        )
    if region.y + region.height > info.height:
        raise ScanError(
            f"region bottom edge {region.y + region.height} exceeds height {info.height}"
        )


def count_red_pixels(
    frame: np.ndarray,
    red_gray_min: int,
    hue_window: int,
    sat_min: int,
) -> int:
    rgb = frame.astype(np.float32)
    red_channel = rgb[..., 0]
    green_channel = rgb[..., 1]
    blue_channel = rgb[..., 2]
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    delta = max_channel - min_channel

    def safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        output = np.zeros_like(numerator)
        np.divide(
            numerator,
            denominator,
            out=output,
            where=denominator != 0,
        )
        return output

    with np.errstate(invalid="ignore"):
        hue = np.where(
            max_channel == red_channel,
            safe_div(green_channel - blue_channel, delta),
            0.0,
        )
        hue = np.where(
            max_channel == green_channel,
            2.0 + safe_div(blue_channel - red_channel, delta),
            hue,
        )
        hue = np.where(
            max_channel == blue_channel,
            4.0 + safe_div(red_channel - green_channel, delta),
            hue,
        )
    hue = np.mod(hue * 60.0, 360.0)

    saturation = delta / np.maximum(max_channel, 1.0)
    saturation_255 = saturation * 255.0
    near_red_hue = (hue <= hue_window) | (hue >= 360.0 - hue_window)
    mask = (
        (frame[..., 0] > red_gray_min)
        & near_red_hue
        & (saturation_255 >= sat_min)
    )
    return int(np.count_nonzero(mask))


def read_exact(stream: object, size: int) -> bytes:
    parts = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def build_ffmpeg_command(
    video: str,
    region: Region,
    sample_rate: float,
    hardware: bool,
) -> List[str]:
    filter_expression = (
        "setpts=PTS-STARTPTS,"
        f"fps={sample_rate:g},"
        "format=rgb24,"
        f"crop={region.width}:{region.height}:{region.x}:{region.y}"
    )
    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    if hardware:
        command += ["-hwaccel", "cuda"]
    command += [
        "-i",
        video,
        "-map",
        "0:v:0",
        "-vf",
        filter_expression,
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    return command


def scan_ffmpeg_stream(
    process: subprocess.Popen,
    region: Region,
    sample_rate: float,
    red_gray_min: int,
    hue_window: int,
    sat_min: int,
    min_pixels: int,
    total_samples: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    snapshot_callback: Optional[SnapshotCallback] = None,
) -> List[Event]:
    frame_bytes = region.width * region.height * 3
    sample_interval = 1.0 / sample_rate
    events: List[Event] = []
    current_start: Optional[float] = None
    current_last: Optional[float] = None
    current_pixels = 0
    sample_number = 0

    def finish_current(reaches_end_of_stream: bool) -> None:
        nonlocal current_start, current_last, current_pixels
        if current_start is None or current_last is None:
            return
        if reaches_end_of_stream:
            duration = current_last - current_start
        else:
            duration = current_last - current_start + sample_interval
        events.append(
            Event(
                start_time=current_start,
                duration=max(0.0, duration),
                red_pixels=current_pixels,
            )
        )
        current_start = None
        current_last = None
        current_pixels = 0

    while True:
        chunk = read_exact(process.stdout, frame_bytes)
        if len(chunk) == 0:
            break
        if len(chunk) != frame_bytes:
            raise ScanError(
                "video stream ended in the middle of a region frame"
            )
        sample_time = sample_number * sample_interval
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
            region.height, region.width, 3
        )
        red_pixels = count_red_pixels(
            frame,
            red_gray_min=red_gray_min,
            hue_window=hue_window,
            sat_min=sat_min,
        )
        if red_pixels >= min_pixels:
            if current_start is None:
                current_start = sample_time
                current_last = sample_time
                current_pixels = red_pixels
                if snapshot_callback is not None:
                    snapshot_callback(sample_time, frame.tobytes())
            else:
                current_last = sample_time
        else:
            finish_current(reaches_end_of_stream=False)

        sample_number += 1
        if progress_callback is not None:
            progress_callback(sample_number, total_samples)
        elif sample_number % 200 == 0:
            print(
                f"Scanned {sample_number} selected frames...",
                file=sys.stderr,
                flush=True,
            )

    finish_current(reaches_end_of_stream=True)
    return events


def decode_once(
    video: str,
    info: VideoInfo,
    region: Region,
    sample_every: int,
    red_gray_min: int,
    hue_window: int,
    sat_min: int,
    min_pixels: int,
    hardware: bool,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    snapshot_callback: Optional[SnapshotCallback] = None,
) -> List[Event]:
    sample_rate = info.fps / sample_every
    total_samples = max(1, int(math.ceil(info.duration * sample_rate)))
    command = build_ffmpeg_command(
        video,
        region,
        sample_rate,
        hardware,
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stderr_lines: List[str] = []

    def drain_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_lines.append(line.decode("utf-8", errors="replace"))

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()
    try:
        assert process.stdout is not None
        events = scan_ffmpeg_stream(
            process,
            region,
            sample_rate,
            red_gray_min,
            hue_window,
            sat_min,
            min_pixels,
            total_samples,
            progress_callback,
            snapshot_callback,
        )
        process.wait()
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        reader.join(timeout=5)

    stderr_text = "".join(stderr_lines).strip()
    if process.returncode != 0:
        suffix = f": {stderr_text}" if stderr_text else ""
        raise ScanError(
            f"ffmpeg exited with code {process.returncode}{suffix}"
        )
    return events


def format_hms(seconds: float, ceil: bool = False) -> str:
    if ceil:
        total = int(math.ceil(max(0.0, seconds)))
    else:
        total = int(math.floor(max(0.0, seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def scan_video(
    video: str,
    output: str,
    region: Region,
    sample_every: int,
    red_gray_min: int,
    hue_window: int,
    sat_min: int,
    min_pixels: int,
    hardware: bool,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    snapshot_callback: Optional[SnapshotCallback] = None,
) -> int:
    info = probe_video(video)
    validate_region(info, region)
    print(
        f"Video: {info.width}x{info.height}, {info.fps:.3f} fps, "
        f"{info.duration:.3f} s, codec {info.codec}",
        file=sys.stderr,
    )

    started = time.monotonic()
    events: Optional[List[Event]] = None
    decoder_name = ""
    if hardware:
        try:
            events = decode_once(
                video,
                info,
                region,
                sample_every,
                red_gray_min,
                hue_window,
                sat_min,
                min_pixels,
                hardware=True,
                progress_callback=progress_callback,
                snapshot_callback=snapshot_callback,
            )
            decoder_name = "CUDA"
        except ScanError as exc:
            print(
                "CUDA decode failed, retrying with CPU: "
                + str(exc),
                file=sys.stderr,
            )
    if events is None:
        events = decode_once(
            video,
            info,
            region,
            sample_every,
            red_gray_min,
            hue_window,
            sat_min,
            min_pixels,
            hardware=False,
            progress_callback=progress_callback,
            snapshot_callback=snapshot_callback,
        )
        decoder_name = "CPU"

    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(
                f"{format_hms(event.start_time)} "
                f"{format_hms(event.duration, ceil=True)} "
                f"red_pixels={event.red_pixels}\n"
            )

    elapsed = time.monotonic() - started
    print(
        f"Finished with {decoder_name}: {len(events)} event(s) in "
        f"{elapsed:.1f} s -> {output}",
        file=sys.stderr,
    )
    return len(events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect red pixels in a region and write event timestamps. "
            "The output line is: HH:MM:SS HH:MM:SS red_pixels=N"
        )
    )
    parser.add_argument("video", help="input mp4 video")
    parser.add_argument(
        "-o",
        "--output",
        help="output txt path (default: beside the video)",
    )
    parser.add_argument(
        "--region",
        type=parse_region,
        default=Region(716, 722, 387, 111),
        help="ROI as x,y,width,height measured from the top-left",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=30,
        help="check every Nth frame (default: 30)",
    )
    parser.add_argument(
        "--red-gray-min",
        "--r-min",
        type=int,
        default=180,
        dest="red_gray_min",
        help=(
            "minimum red-channel grayscale value 0-255 "
            "(default: 180)"
        ),
    )
    parser.add_argument(
        "--hue-window",
        type=int,
        default=10,
        help="degrees from red hue 0/360 considered red (default: 10)",
    )
    parser.add_argument(
        "--sat-min",
        type=int,
        default=60,
        help="minimum HSV saturation 0-255 (default: 60)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=400,
        help="minimum red pixels required to record an event (default: 400)",
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="disable CUDA hardware decoding",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_every <= 0:
        print("--sample-every must be positive", file=sys.stderr)
        return 2
    if not (0 <= args.red_gray_min <= 255):
        print("--red-gray-min must be between 0 and 255", file=sys.stderr)
        return 2
    if not (0 <= args.hue_window <= 180):
        print("--hue-window must be between 0 and 180", file=sys.stderr)
        return 2
    if not (0 <= args.sat_min <= 255):
        print("--sat-min must be between 0 and 255", file=sys.stderr)
        return 2
    if args.min_pixels < 1:
        print("--min-pixels must be positive", file=sys.stderr)
        return 2

    if args.output is None:
        directory = os.path.dirname(args.video)
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(directory, f"{stem}.red_timestamps.txt")

    try:
        scan_video(
            args.video,
            args.output,
            args.region,
            args.sample_every,
            args.red_gray_min,
            args.hue_window,
            args.sat_min,
            args.min_pixels,
            hardware=not args.no_hardware,
        )
    except ScanError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
