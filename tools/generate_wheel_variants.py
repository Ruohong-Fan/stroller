#!/usr/bin/env python3
"""Generate the missing width/wheel GLB variants from the existing CAD scenes.

The source repository contains six assembled postures for each frame width,
with one default wheelset per width (380/S, 550/M, 650/L). This script keeps
the active scene from each source GLB, scales only the four wheel assemblies
to the requested wheelset dimensions, and writes compact GLB files.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import struct
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "no_cabin"
DEFAULT_WHEEL = {380: "S", 550: "M", 650: "L"}
SPECS = {
    "S": {"front_dia": 150, "front_width": 25, "rear_dia": 200, "rear_width": 30},
    "M": {"front_dia": 220, "front_width": 35, "rear_dia": 300, "rear_width": 45},
    "L": {"front_dia": 260, "front_width": 45, "rear_dia": 360, "rear_width": 60},
}
MISSING = {
    380: ("M", "L"),
    550: ("S", "L"),
    650: ("S", "M"),
}


def read_glb(path: pathlib.Path) -> tuple[dict, bytearray]:
    raw = path.read_bytes()
    magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(raw):
        raise ValueError(f"Invalid GLB header: {path}")
    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset < len(raw):
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_chunk = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = bytearray(chunk)
    if json_chunk is None or bin_chunk is None:
        raise ValueError(f"GLB is missing a required chunk: {path}")
    return json_chunk, bin_chunk


def accessor_bytes(doc: dict, binary: bytearray, accessor_index: int) -> tuple[int, int, int]:
    accessor = doc["accessors"][accessor_index]
    view = doc["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 12)
    return start, stride, accessor["count"]


def read_positions(doc: dict, binary: bytearray, accessor_index: int) -> list[tuple[float, float, float]]:
    accessor = doc["accessors"][accessor_index]
    if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3":
        raise ValueError("Wheel position data must be float VEC3")
    start, stride, count = accessor_bytes(doc, binary, accessor_index)
    return [struct.unpack_from("<3f", binary, start + i * stride) for i in range(count)]


def transform_vec3(doc: dict, binary: bytearray, accessor_index: int, fn) -> list[tuple[float, float, float]]:
    accessor = doc["accessors"][accessor_index]
    if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3":
        raise ValueError("Wheel data must be float VEC3")
    start, stride, count = accessor_bytes(doc, binary, accessor_index)
    values = []
    for i in range(count):
        pos = start + i * stride
        value = fn(struct.unpack_from("<3f", binary, pos))
        struct.pack_into("<3f", binary, pos, *value)
        values.append(value)
    return values


def position_accessor_for_node(doc: dict, node: dict) -> int:
    primitive = doc["meshes"][node["mesh"]]["primitives"][0]
    return primitive["attributes"]["POSITION"]


def scene_wheel_centers(doc: dict, binary: bytearray, active_nodes: list[int]) -> dict[str, tuple[float, float, float]]:
    centers = {}
    for wheel in ("FL", "FR", "RL", "RR"):
        tire = next(
            (doc["nodes"][i] for i in active_nodes if doc["nodes"][i].get("name", "").startswith(wheel + "_tire")),
            None,
        )
        if tire is None:
            raise ValueError(f"Active scene has no {wheel} tire")
        accessor_index = position_accessor_for_node(doc, tire)
        values = read_positions(doc, binary, accessor_index)
        mins = [min(v[a] for v in values) for a in range(3)]
        maxs = [max(v[a] for v in values) for a in range(3)]
        centers[wheel] = tuple((mins[a] + maxs[a]) / 2 for a in range(3))
    return centers


def apply_wheel_dimensions(doc: dict, binary: bytearray, active_nodes: list[int], source: str, target: str) -> None:
    if source == target:
        return
    old = SPECS[source]
    new = SPECS[target]
    centers = scene_wheel_centers(doc, binary, active_nodes)
    modified_positions = set()
    modified_normals = set()

    for node_index in active_nodes:
        node = doc["nodes"][node_index]
        name = node.get("name", "")
        wheel = next((w for w in ("FL", "FR", "RL", "RR") if name.startswith(w + "_")), None)
        if wheel is None or "mesh" not in node:
            continue
        front = wheel in ("FL", "FR")
        diameter_ratio = (new["front_dia"] / old["front_dia"]) if front else (new["rear_dia"] / old["rear_dia"])
        width_ratio = (new["front_width"] / old["front_width"]) if front else (new["rear_width"] / old["rear_width"])
        cx, cy, cz = centers[wheel]
        mesh = doc["meshes"][node["mesh"]]
        for primitive in mesh["primitives"]:
            attrs = primitive["attributes"]
            pos_index = attrs["POSITION"]
            if pos_index not in modified_positions:
                def scale_position(p):
                    return (
                        cx + (p[0] - cx) * diameter_ratio,
                        cy + (p[1] - cy) * diameter_ratio,
                        cz + (p[2] - cz) * width_ratio,
                    )

                positions = transform_vec3(doc, binary, pos_index, scale_position)
                doc["accessors"][pos_index]["min"] = [min(p[a] for p in positions) for a in range(3)]
                doc["accessors"][pos_index]["max"] = [max(p[a] for p in positions) for a in range(3)]
                modified_positions.add(pos_index)
            normal_index = attrs.get("NORMAL")
            if normal_index is not None and normal_index not in modified_normals:
                def scale_normal(n):
                    nx, ny, nz = n[0] / diameter_ratio, n[1] / diameter_ratio, n[2] / width_ratio
                    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                    return (nx / length, ny / length, nz / length)

                transform_vec3(doc, binary, normal_index, scale_normal)
                modified_normals.add(normal_index)


def compact_active_scene(doc: dict, binary: bytearray) -> tuple[dict, bytearray]:
    active_scene = doc.get("scene", 0)
    roots = doc["scenes"][active_scene]["nodes"]
    if any(doc["nodes"][i].get("children") for i in roots):
        raise ValueError("Unexpected child hierarchy in CAD scene")

    used_meshes = sorted({doc["nodes"][i]["mesh"] for i in roots if "mesh" in doc["nodes"][i]})
    mesh_map = {old: new for new, old in enumerate(used_meshes)}
    used_accessors = set()
    used_materials = set()
    for mesh_index in used_meshes:
        for primitive in doc["meshes"][mesh_index]["primitives"]:
            used_accessors.update(primitive["attributes"].values())
            if "indices" in primitive:
                used_accessors.add(primitive["indices"])
            if "material" in primitive:
                used_materials.add(primitive["material"])
            for target in primitive.get("targets", []):
                used_accessors.update(target.values())
    accessor_map = {old: new for new, old in enumerate(sorted(used_accessors))}
    used_views = set()
    for accessor_index in used_accessors:
        accessor = doc["accessors"][accessor_index]
        if accessor.get("sparse"):
            raise ValueError("Sparse accessors are not supported by the GLB compactor")
        if "bufferView" in accessor:
            used_views.add(accessor["bufferView"])
    view_map = {old: new for new, old in enumerate(sorted(used_views))}

    compact_bin = bytearray()
    new_views = []
    for old_index in sorted(used_views):
        view = copy.deepcopy(doc["bufferViews"][old_index])
        start = view.get("byteOffset", 0)
        length = view["byteLength"]
        while len(compact_bin) % 4:
            compact_bin.append(0)
        view["byteOffset"] = len(compact_bin)
        view["buffer"] = 0
        compact_bin.extend(binary[start : start + length])
        new_views.append(view)

    new_accessors = []
    for old_index in sorted(used_accessors):
        accessor = copy.deepcopy(doc["accessors"][old_index])
        if "bufferView" in accessor:
            accessor["bufferView"] = view_map[accessor["bufferView"]]
        new_accessors.append(accessor)

    new_meshes = []
    for old_index in used_meshes:
        mesh = copy.deepcopy(doc["meshes"][old_index])
        for primitive in mesh["primitives"]:
            primitive["attributes"] = {k: accessor_map[v] for k, v in primitive["attributes"].items()}
            if "indices" in primitive:
                primitive["indices"] = accessor_map[primitive["indices"]]
            if "material" in primitive:
                primitive["material"] = primitive["material"]
            for target in primitive.get("targets", []):
                for key, value in target.items():
                    target[key] = accessor_map[value]
        new_meshes.append(mesh)

    new_materials = [copy.deepcopy(doc["materials"][i]) for i in sorted(used_materials)]
    material_map = {old: new for new, old in enumerate(sorted(used_materials))}
    for mesh in new_meshes:
        for primitive in mesh["primitives"]:
            if "material" in primitive:
                primitive["material"] = material_map[primitive["material"]]

    new_nodes = []
    node_map = {}
    for old_index in roots:
        node = copy.deepcopy(doc["nodes"][old_index])
        if "mesh" in node:
            node["mesh"] = mesh_map[node["mesh"]]
        node_map[old_index] = len(new_nodes)
        new_nodes.append(node)

    out = copy.deepcopy(doc)
    out["nodes"] = new_nodes
    out["meshes"] = new_meshes
    out["accessors"] = new_accessors
    out["bufferViews"] = new_views
    out["buffers"] = [{"byteLength": len(compact_bin)}]
    out["materials"] = new_materials
    out["scenes"] = [{"nodes": list(range(len(new_nodes)))}]
    out["scene"] = 0
    # The source scenes use only mesh nodes, no animations, textures, skins or cameras.
    for key in ("animations", "textures", "images", "samplers", "skins", "cameras"):
        out.pop(key, None)
    out.pop("extensionsUsed", None)
    out.pop("extensionsRequired", None)
    out.pop("extensions", None)
    return out, compact_bin


def write_glb(path: pathlib.Path, doc: dict, binary: bytearray) -> None:
    doc_bytes = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    while len(doc_bytes) % 4:
        doc_bytes += b" "
    while len(binary) % 4:
        binary.append(0)
    total = 12 + 8 + len(doc_bytes) + 8 + len(binary)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(doc_bytes), 0x4E4F534A))
        f.write(doc_bytes)
        f.write(struct.pack("<II", len(binary), 0x004E4942))
        f.write(binary)


def generate(width: int, state: int, target_wheel: str) -> pathlib.Path:
    source_wheel = DEFAULT_WHEEL[width]
    source_path = MODEL_DIR / f"w{width}_state{state}_wheel_{source_wheel}.glb"
    target_path = MODEL_DIR / f"w{width}_state{state}_wheel_{target_wheel}.glb"
    doc, binary = read_glb(source_path)
    active_nodes = list(doc["scenes"][doc.get("scene", 0)]["nodes"])
    apply_wheel_dimensions(doc, binary, active_nodes, source_wheel, target_wheel)
    compact_doc, compact_binary = compact_active_scene(doc, binary)
    write_glb(target_path, compact_doc, compact_binary)
    return target_path


def main() -> int:
    generated = []
    for width, wheelsets in MISSING.items():
        for wheel in wheelsets:
            for state in range(1, 7):
                generated.append(generate(width, state, wheel))
    total = sum(path.stat().st_size for path in generated)
    print(f"Generated {len(generated)} GLB files ({total / 1024 / 1024:.1f} MiB total)")
    for path in generated:
        print(f"{path.relative_to(ROOT)}\t{path.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
