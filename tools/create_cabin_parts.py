#!/usr/bin/env python3
"""Export the separate cabin modules and their existing catalogue accessories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from catalog_cabin_geometry import rotate_vec
from catalog_modeling import group
from create_cabin_catalog import load_catalog
from create_d02_double_seat import ROOT, mesh_node
from generate_wheel_variants import DEFAULT_WHEEL, compact_active_scene, read_glb, write_glb


def single_scene(cabin_id, width):
    wheel = DEFAULT_WHEEL[width]
    name = f"{cabin_id}_w{width}_state1_wheel_{wheel}.glb"
    return read_glb(ROOT / "models/single_cabin" / name)


def isolate_module(doc, binary, names):
    """Remove the chassis and undo its common leveling angle."""
    scene = doc["scenes"][doc.get("scene", 0)]
    roots = [index for index in scene["nodes"] if doc["nodes"][index].get("name") in names]
    if len(roots) != len(names) or not roots:
        raise ValueError(f"Could not locate the complete module: {sorted(names)}")
    scene["nodes"] = roots
    part, part_bin = compact_active_scene(doc, binary)
    first = part["nodes"][0]
    origin = first.get("translation", (0, 0, 0))
    attitude = first.get("rotation", (0, 0, 0, 1))
    inverse = tuple(-value for value in attitude[:3]) + (attitude[3],)
    lower, upper = [math.inf] * 3, [-math.inf] * 3
    for node in part["nodes"]:
        angle = node.pop("rotation", (0, 0, 0, 1))
        if list(angle) != list(attitude):
            raise ValueError("Module components must share one assembly angle")
        position = node.get("translation", (0, 0, 0))
        local = rotate_vec(inverse, tuple(a - b for a, b in zip(position, origin)))
        node["translation"] = local
        for primitive in part["meshes"][node["mesh"]]["primitives"]:
            accessor = part["accessors"][primitive["attributes"]["POSITION"]]
            for axis in range(3):
                lower[axis] = min(lower[axis], local[axis] + accessor["min"][axis])
                upper[axis] = max(upper[axis], local[axis] + accessor["max"][axis])
    # The centered part has a useful pivot even in GLB viewers outside the site.
    center = tuple((a + b) / 2 for a, b in zip(lower, upper))
    for node in part["nodes"]:
        node["translation"] = tuple(a - b for a, b in zip(node["translation"], center))
    return part, part_bin


def cabin_module(cabin_id, width):
    doc, binary = single_scene(cabin_id, width)
    prefix = f"{cabin_id}_Cabin"
    names = {
        node["name"] for node in doc["nodes"]
        if node.get("name", "").startswith(prefix) and not node["name"].startswith(prefix + "_Footrest")
    }
    return isolate_module(doc, binary, names)


def adult_footrest(cabin_id, width):
    doc, binary = single_scene(cabin_id, width)
    prefix = f"{cabin_id}_Cabin_Footrest"
    return isolate_module(doc, binary, {prefix, prefix + "_LowDocks"})


def cargo_tray(name, length, width):
    """Fit a tabletop and four open-bottom clips to the cargo rim."""
    parts = group("cargo_edge", "shell", "mounts", "guard_release")
    parts["cargo_edge"].seat_pad((0, 0.025, 0), (length + 0.028, 0.024, width + 0.028),
                                0.061, 0.007, 0)
    for z in (-width * 0.29, 0, width * 0.29):
        parts["shell"].box((0, 0.010, z), (length * 0.86, 0.012, 0.027))
    angles = [math.radians(-108 + 216 * i / 16) for i in range(17)]
    for x in (-length * 0.32, length * 0.32):
        for side in (-1, 1):
            z = side * width / 2
            jaws = parts["mounts"]
            sections = [
                [(x + dx, -0.004 + r * math.cos(angle), z + r * math.sin(angle)) for angle in angles]
                for dx, r in ((-0.020, 0.022), (0.020, 0.022), (0.020, 0.012), (-0.020, 0.012))
            ]
            for a, b in zip(sections, [*sections[1:], sections[0]]):
                for i in range(len(angles) - 1):
                    jaws.quad(a[i], a[i + 1], b[i + 1], b[i])
            jaws.quad(*(section[0] for section in sections))
            jaws.quad(*(section[-1] for section in reversed(sections)))
            parts["guard_release"].box((x, 0.004, z + side * 0.018), (0.025, 0.012, 0.011))
    doc = {
        "asset": {"version": "2.0", "generator": "Stroller cabin parts"},
        "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
        "materials": [], "accessors": [], "bufferViews": [],
    }
    binary = bytearray()
    doc["scenes"][0]["nodes"].append(mesh_node(doc, binary, name, parts))
    doc["buffers"] = [{"byteLength": len(binary)}]
    return doc, binary


def build_cabin_parts(output):
    """Write the discrete assets and return their site catalogue entries."""
    index = ROOT / "index.html"
    labels = dict(re.findall(r"\bid:\s*'(S\d{2})',\s*name:\s*'([^']+)'", index.read_text()))
    folder = Path(output) / "other/cabins"
    folder.mkdir(parents=True, exist_ok=True)
    entries = []

    def export(part_id, name, model, description=None):
        doc, binary = model
        write_glb(folder / f"{part_id}.glb", doc, binary)
        entry = {"id": part_id, "name": name, "path": f"other/cabins/{part_id}"}
        if description:
            entry["description"] = description
        entries.append(entry)

    for mode, spec, _ in load_catalog(index):
        if mode != "single":
            continue
        cabin_id = spec["id"]
        width = spec["widths"][0]
        if cabin_id == "S11":
            for width in spec["widths"]:
                label = f"{labels[cabin_id]} · {width} mm"
                export(f"{cabin_id}-{width}", label, cabin_module(cabin_id, width))
        else:
            export(cabin_id, labels[cabin_id], cabin_module(cabin_id, width))

    for part_id, cabin_id, name, width in (
        ("FP08", "S08", "中型成人座舱脚踏", 550),
        ("FP09", "S09", "大型成人座舱脚踏", 650),
    ):
        export(part_id, name, adult_footrest(cabin_id, width), f"配合 {cabin_id} 使用，卡接在车架低位。")
    for part_id, name, length, width, cabin in (
        ("A01", "中型载货舱桌板", 0.600, 0.450, "S13"),
        ("A02", "大型载货舱桌板", 0.670, 0.546, "S14"),
    ):
        export(part_id, name, cargo_tray(part_id, length, width), f"桌板通过底部卡扣固定在 {cabin} 上沿。")
    for part_id, label, side in (("AD650-L", "左侧", -1), ("AD650-R", "右侧", 1)):
        doc, binary = single_scene("S05", 650)
        adapter = isolate_module(doc, binary, {f"D02_WideAdapter_M_{side}"})
        export(part_id, f"宽档座舱转接件 · {label}", adapter,
               "用于将标准接口的单儿童坐席、婴儿睡篮或婴儿提篮安装到宽档车架。")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    print(json.dumps(build_cabin_parts(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
