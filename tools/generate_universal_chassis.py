#!/usr/bin/env python3
"""Fit reversible cabin receivers to the bare chassis display models."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_wheel_variants import DEFAULT_WHEEL, MODEL_DIR, SPECS, compact_active_scene, read_glb, write_glb
from universal_mount import standardize_receivers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=tuple(DEFAULT_WHEEL), action="append")
    parser.add_argument("--state", type=int, choices=range(1, 7), action="append")
    parser.add_argument("--wheel", choices=tuple(SPECS), action="append")
    args = parser.parse_args()
    widths = args.width or DEFAULT_WHEEL
    states = args.state or range(1, 7)
    wheels = args.wheel or SPECS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with TemporaryDirectory(dir=args.output_dir, prefix=".universal-chassis-") as work:
        for width in widths:
            reference = read_glb(args.source_dir / f"w{width}_state1_wheel_{DEFAULT_WHEEL[width]}.glb")
            for state in states:
                for wheel in wheels:
                    name = f"w{width}_state{state}_wheel_{wheel}.glb"
                    doc, binary = read_glb(args.source_dir / name)
                    standardize_receivers(doc, binary, reference=reference)
                    doc, binary = compact_active_scene(doc, binary)
                    staged = Path(work) / name
                    write_glb(staged, doc, binary)
                    staged.replace(args.output_dir / name)
                    count += 1
    print(f"Exported {count} chassis models to {args.output_dir}")


if __name__ == "__main__":
    main()
