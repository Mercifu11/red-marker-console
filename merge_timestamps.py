#!/usr/bin/env python3
"""Merge marker timestamps that fall within a fixed gap of a segment start."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


def parse_hms(text: str) -> float:
    parts = text.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(text)


def format_hms(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_times(path: str) -> List[float]:
    times = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            times.append(parse_hms(line.split()[0]))
    return times


def merge_times(times: List[float], gap: float) -> List[float]:
    ordered = sorted(times)
    groups: List[List[float]] = []
    for value in ordered:
        if groups and value - groups[-1][0] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [group[0] for group in groups]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge nearby red marker timestamps into segment starts."
    )
    parser.add_argument("markers", help="source red timestamp txt")
    parser.add_argument(
        "--gap",
        type=float,
        default=10.0,
        help="markers within this many seconds of a segment start merge",
    )
    parser.add_argument("-o", "--output", help="merged txt output path")
    args = parser.parse_args(argv)

    times = read_times(args.markers)
    merged = merge_times(times, args.gap)
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.markers))[0]
        args.output = os.path.join(
            os.path.dirname(args.markers),
            f"{stem}.grouped_{args.gap:g}s.txt",
        )

    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"# merged {len(times)} markers -> {len(merged)} groups "
            f"(gap <= {args.gap:g}s)\n"
        )
        for value in merged:
            handle.write(format_hms(value) + "\n")

    print(
        f"Merged {len(times)} markers into {len(merged)} groups -> "
        + args.output,
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
