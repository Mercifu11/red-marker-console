#!/usr/bin/env python3
"""Create a short synthetic mp4 with known red, orange, and white blocks."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def find_ffmpeg() -> str:
    configured = os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        default="test_red_events.mp4",
        help="output mp4 path",
    )
    args = parser.parse_args(argv)

    command = [
        find_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1920x1080:r=60:d=10",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=300x80:r=60:d=10",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=200x60:r=60:d=10",
        "-f",
        "lavfi",
        "-i",
        "color=c=orange:s=50x50:r=60:d=10",
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=60x60:r=60:d=10",
        "-filter_complex",
        (
            "[0:v][1:v]overlay=x=730:y=780:enable='between(t,1,3)'[v1];"
            "[v1][2:v]overlay=x=750:y=790:enable='between(t,5.5,7)'[v2];"
            "[v2][3:v]overlay=x=900:y=820:enable='between(t,4,5)'[v3];"
            "[v3][4:v]overlay=x=760:y=790:enable='between(t,8,9)'[vout]"
        ),
        "-map",
        "[vout]",
        "-t",
        "10",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        args.output,
    ]
    print("Running:", " ".join(command), file=sys.stderr)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("ffmpeg failed", file=sys.stderr)
        return result.returncode
    print(f"Created {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
