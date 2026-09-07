#!/usr/bin/env python3
"""Cut marker-centered video segments and export every audio track.

For each marker in the red-timestamp txt, this creates:

  <output>/1.mp4, <output>/2.mp4, ...
  <output>/1-1.m4a, <output>/1-2.m4a, ... (audio track per video segment)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import List, Tuple


def find_ffmpeg() -> str:
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def find_ffprobe() -> str:
    configured = os.environ.get("FFPROBE_BIN")
    if configured:
        return configured
    found = shutil.which("ffprobe")
    return found or "ffprobe"


def parse_hms(text: str) -> float:
    text = text.strip()
    parts = text.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(text)


def probe_media(video: str) -> Tuple[float, List[int]]:
    command = [
        find_ffprobe(),
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type:format=duration",
        "-of",
        "json",
        video,
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
        raise RuntimeError("ffprobe failed: " + result.stderr.strip())
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    audio_indices = [
        int(stream["index"])
        for stream in streams
        if stream.get("codec_type") == "audio"
    ]
    duration = float((data.get("format") or {}).get("duration", 0))
    return duration, audio_indices


def read_markers(path: str) -> List[float]:
    markers = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            markers.append(parse_hms(line.split()[0]))
    return markers


def make_ranges(
    markers: List[float],
    duration: float,
    before: float,
    after: float,
) -> List[Tuple[float, float]]:
    ranges = []
    for marker in markers:
        start = max(0.0, marker - before)
        end = min(duration, marker + after)
        if end <= start:
            continue
        ranges.append((start, end))
    return ranges


def run_command(command: List[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed: " + " ".join(command) + f" ({result.returncode})"
        )


def export_segment(
    video: str,
    output_dir: str,
    segment_number: int,
    start: float,
    end: float,
    audio_indices: List[int],
    audio_output_numbers: List[int] | None = None,
) -> None:
    duration = end - start
    start_text = f"{start:.3f}"
    duration_text = f"{duration:.3f}"

    video_output = os.path.join(output_dir, f"{segment_number}.mp4")
    command = [
        find_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        start_text,
        "-t",
        duration_text,
        "-i",
        video,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        video_output,
    ]

    if audio_output_numbers is None:
        audio_output_numbers = list(range(1, len(audio_indices) + 1))
    for audio_number, stream_index in zip(audio_output_numbers, audio_indices):
        audio_output = os.path.join(
            output_dir, f"{segment_number}-{audio_number}.m4a"
        )
        command += [
            "-map",
            f"0:{stream_index}",
            "-c:a",
            "copy",
            "-vn",
            "-sn",
            "-dn",
            audio_output,
        ]
    run_command(command)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read red timestamps, cut marker +/- window video segments, "
            "and export every audio track for each segment."
        )
    )
    parser.add_argument("video", help="source mp4 video")
    parser.add_argument(
        "--markers",
        help=(
            "txt with marker times; "
            "defaults to <video>.red_timestamps.txt"
        ),
    )
    parser.add_argument(
        "--before",
        type=float,
        default=10.0,
        help="seconds before each marker (default: 10)",
    )
    parser.add_argument(
        "--after",
        type=float,
        default=10.0,
        help="seconds after each marker (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        help="output folder (default: <video directory>/<video name>)",
    )
    parser.add_argument(
        "--audio-tracks",
        help=(
            "comma-separated 1-based audio positions to export, "
            "for example 1,3,4; default exports all"
        ),
    )
    args = parser.parse_args(argv)

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    video_dir = os.path.dirname(args.video)
    if args.markers is None:
        args.markers = os.path.join(video_dir, f"{video_stem}.red_timestamps.txt")
    if args.output_dir is None:
        args.output_dir = os.path.join(video_dir, video_stem)
    os.makedirs(args.output_dir, exist_ok=True)

    markers = read_markers(args.markers)
    if not markers:
        print("No markers found in " + args.markers, file=sys.stderr)
        return 2

    duration, audio_indices = probe_media(args.video)
    audio_output_numbers = None
    if args.audio_tracks:
        try:
            positions = [
                int(part.strip())
                for part in args.audio_tracks.split(",")
                if part.strip()
            ]
        except ValueError as exc:
            raise RuntimeError("--audio-tracks must contain integers") from exc
        selected_pairs = [
            (position, audio_indices[position - 1])
            for position in positions
            if 1 <= position <= len(audio_indices)
        ]
        audio_indices = [
            audio_indices[position - 1]
            for position in positions
            if 1 <= position <= len(audio_indices)
        ]
        audio_output_numbers = [position for position, _ in selected_pairs]
        if not audio_indices:
            raise RuntimeError("no selected audio tracks remain valid")
    ranges = make_ranges(markers, duration, args.before, args.after)
    if not ranges:
        print("No valid ranges remain", file=sys.stderr)
        return 2

    print(
        f"Video duration: {duration:.3f}s, audio tracks: "
        f"{len(audio_indices)}",
        file=sys.stderr,
    )
    print(
        f"Output folder: {args.output_dir}",
        file=sys.stderr,
    )
    for index, (start, end) in enumerate(ranges, start=1):
        print(
            f"Segment {index}: {start:.3f} - {end:.3f} "
            f"({end - start:.3f}s)",
            file=sys.stderr,
        )

    for index, (start, end) in enumerate(ranges, start=1):
        print(
            f"Exporting segment {index}/{len(ranges)}: "
            f"{start:.3f} - {end:.3f}",
            file=sys.stderr,
            flush=True,
        )
        export_segment(
            args.video,
            args.output_dir,
            index,
            start,
            end,
            audio_indices,
            audio_output_numbers,
        )

    print("Finished exporting segments", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
