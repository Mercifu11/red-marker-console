#!/usr/bin/env python3
"""Cut marker-centered segments from a video and join them in order."""

from __future__ import annotations

import argparse
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


def probe_duration(video: str) -> float:
    command = [
        find_ffprobe(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
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
    return float(result.stdout.strip())


def has_audio(video: str) -> bool:
    command = [
        find_ffprobe(),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
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
    return result.returncode == 0 and result.stdout.strip() != ""


def read_markers(path: str) -> List[float]:
    markers = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split()[0]
            markers.append(parse_hms(first))
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


def build_ffmpeg_command(
    video: str,
    output: str,
    ranges: List[Tuple[float, float]],
    include_audio: bool,
) -> List[str]:
    filters = []
    for index, (start, end) in enumerate(ranges):
        filters.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        if include_audio:
            filters.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )

    count = len(ranges)
    if include_audio:
        segment_chain = "".join(f"[v{i}][a{i}]" for i in range(count))
        filters.append(
            f"{segment_chain}"
            f"concat=n={count}:v=1:a=1[vout][aout]"
        )
    else:
        video_chain = "".join(f"[v{i}]" for i in range(count))
        filters.append(f"{video_chain}concat=n={count}:v=1:a=0[vout]")

    command = [
        find_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
    ]
    if include_audio:
        command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    command += [
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
        output,
    ]
    return command


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read marker times from a txt file, cut marker +/- window "
            "segments, and join them in order."
        )
    )
    parser.add_argument("video", help="source mp4 video")
    parser.add_argument(
        "--markers",
        help=(
            "txt with marker times, one per line; "
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
        "-o",
        "--output",
        help="joined output path (default: beside the video)",
    )
    args = parser.parse_args(argv)

    if args.markers is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.markers = os.path.join(
            os.path.dirname(args.video), f"{stem}.red_timestamps.txt"
        )
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(
            os.path.dirname(args.video), f"{stem}_joined.mp4"
        )

    markers = read_markers(args.markers)
    if not markers:
        print("No markers found in " + args.markers, file=sys.stderr)
        return 2

    duration = probe_duration(args.video)
    ranges = make_ranges(markers, duration, args.before, args.after)
    if not ranges:
        print("No valid ranges remain", file=sys.stderr)
        return 2

    include_audio = has_audio(args.video)
    print("Duration: %.3f s" % duration, file=sys.stderr)
    for index, (start, end) in enumerate(ranges):
        print(
            "Segment %d: %.3f - %.3f (%.3f s)"
            % (index + 1, start, end, end - start),
            file=sys.stderr,
        )

    command = build_ffmpeg_command(
        args.video,
        args.output,
        ranges,
        include_audio,
    )
    print("Running ffmpeg...", file=sys.stderr)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("ffmpeg failed", file=sys.stderr)
        return result.returncode
    print("Created " + args.output, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
