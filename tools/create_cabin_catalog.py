#!/usr/bin/env python3
"""Build the assembled GLB for every cabin, supported width and wheelset."""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import cache
import json
from pathlib import Path
import re

from catalog_frame import prepare_catalog_frame
from create_d02_double_seat import ROOT
from double_cabin_catalog import build_double
from generate_wheel_variants import SPECS as WHEELSETS, write_glb
from single_cabin_catalog import build_single


def load_catalog(index_path):
    """Use the published selector as the source for model IDs and compatibility."""
    page = index_path.read_text()
    for mode, builder, expected in (("single", build_single, 15), ("double", build_double, 35)):
        block = re.search(rf"const {mode.upper()}_CABINS = \[\n(.*?)\n\];", page, re.S)
        if not block:
            raise ValueError(f"The {mode} cabin selector is missing")
        rows = re.findall(r"^\s*\{(.+)\}\s*,?\s*$", block[1], re.M)
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} {mode} cabin rows, found {len(rows)}")
        for row in rows:
            spec = dict(re.findall(r"\b(id|lower|upper): '([SD]\d{2})'", row))
            spec["widths"] = json.loads(re.search(r"\bwidths: (\[[\d, ]+\])", row)[1])
            spec["ports"] = re.findall(r"'(LOW|MID|HIGH)'", re.search(r"\bports: (\[[^\]]+\])", row)[1])
            if spec["id"] != "D02":
                yield mode, spec, builder


def build_catalog(output, *, cabins=(), wheels=(), widths=()):
    prepare_frame = cache(prepare_catalog_frame)
    built = []
    for mode, spec, builder in load_catalog(ROOT / "index.html"):
        cabin_id = spec["id"]
        if cabins and cabin_id not in cabins:
            continue
        folder = output / f"{mode}_cabin"
        folder.mkdir(parents=True, exist_ok=True)
        for width in spec["widths"]:
            if widths and width not in widths:
                continue
            for wheel in wheels or WHEELSETS:
                ports = tuple(port[0] for port in spec["ports"])
                frame, frame_binary, centers, level = prepare_frame(width, wheel, ports)
                doc, binary = deepcopy(frame), bytearray(frame_binary)
                builder(doc, binary, spec, width, centers, level)
                file = folder / f"{cabin_id}_w{width}_state1_wheel_{wheel}.glb"
                write_glb(file, doc, binary)
                built.append(file)
                print(f"{file.relative_to(output)} ({file.stat().st_size:,} bytes)", flush=True)
    if not built:
        raise ValueError("No catalogue entry matched the selected filters")
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--cabin", action="append")
    parser.add_argument("--wheel", choices=tuple(WHEELSETS), action="append")
    parser.add_argument("--width", type=int, choices=(380, 550, 650), action="append")
    args = parser.parse_args()
    built = build_catalog(args.output, cabins=args.cabin, wheels=args.wheel, widths=args.width)
    print(f"Generated {len(built)} assembled previews", flush=True)


if __name__ == "__main__":
    main()
