#!/usr/bin/env python3
"""Build a first D02 double-child-seat concept GLB from the 550/M chassis."""

from __future__ import annotations

import json
import math
import pathlib
import struct

from generate_wheel_variants import compact_active_scene


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/no_cabin/w550_state1_wheel_M.glb"
OUTPUT = ROOT / "models/double_cabin/D02_w550_state1_wheel_M.glb"


def read_glb(path: pathlib.Path) -> tuple[dict, bytearray]:
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or length != len(raw):
        raise ValueError(f"Invalid source GLB: {path}")
    offset, doc, binary = 12, None, None
    while offset < len(raw):
        size, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + size]
        offset += size
        if chunk_type == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = bytearray(chunk)
    if doc is None or binary is None:
        raise ValueError("Source GLB is missing JSON or BIN chunk")
    return doc, binary


def write_glb(path: pathlib.Path, doc: dict, binary: bytearray) -> None:
    while len(binary) % 4:
        binary.append(0)
    doc["buffers"][0]["byteLength"] = len(binary)
    json_chunk = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        f.write(json_chunk)
        f.write(struct.pack("<II", len(binary), 0x004E4942))
        f.write(binary)


def unit(v):
    length = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / length for x in v)


def sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class Geometry:
    def __init__(self):
        self.vertices: list[tuple[float, float, float]] = []
        self.normals: list[tuple[float, float, float]] = []

    def triangle(self, a, b, c):
        normal = unit(cross(sub(b, a), sub(c, a)))
        self.vertices.extend((a, b, c))
        self.normals.extend((normal, normal, normal))

    def quad(self, a, b, c, d):
        self.triangle(a, b, c)
        self.triangle(a, c, d)

    def side_cheek(self, points, z_center, thickness):
        """Extrude a triangular molded cheek in the seat's side plane."""
        z0, z1 = z_center - thickness / 2, z_center + thickness / 2
        front = [(x, y, z0) for x, y in points]
        back = [(x, y, z1) for x, y in points]
        self.triangle(*front)
        self.triangle(*reversed(back))
        for i in range(3):
            j = (i + 1) % 3
            self.quad(front[i], front[j], back[j], back[i])

    def box(self, center, dims, angle=0.0):
        cx, cy, cz = center
        dx, dy, dz = (v / 2 for v in dims)
        co, si = math.cos(angle), math.sin(angle)

        def point(x, y, z):
            return (cx + x * co - y * si, cy + x * si + y * co, cz + z)

        p = {(x, y, z): point(x * dx, y * dy, z * dz) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)}
        faces = [
            [p[-1, -1, -1], p[-1, -1, 1], p[-1, 1, 1], p[-1, 1, -1]],
            [p[1, -1, -1], p[1, 1, -1], p[1, 1, 1], p[1, -1, 1]],
            [p[-1, -1, -1], p[1, -1, -1], p[1, -1, 1], p[-1, -1, 1]],
            [p[-1, 1, -1], p[-1, 1, 1], p[1, 1, 1], p[1, 1, -1]],
            [p[-1, -1, -1], p[-1, 1, -1], p[1, 1, -1], p[1, -1, -1]],
            [p[-1, -1, 1], p[1, -1, 1], p[1, 1, 1], p[-1, 1, 1]],
        ]
        for face in faces:
            self.quad(*face)

    def beam(self, start, end, radius, sides=12):
        axis = unit(sub(end, start))
        ref = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (0.0, 1.0, 0.0)
        u = unit(cross(axis, ref))
        v = unit(cross(axis, u))
        ring_a, ring_b = [], []
        for i in range(sides):
            a = 2 * math.pi * i / sides
            radial = tuple(u[j] * math.cos(a) + v[j] * math.sin(a) for j in range(3))
            ring_a.append(tuple(start[j] + radial[j] * radius for j in range(3)))
            ring_b.append(tuple(end[j] + radial[j] * radius for j in range(3)))
        for i in range(sides):
            j = (i + 1) % sides
            self.quad(ring_a[i], ring_a[j], ring_b[j], ring_b[i])
            self.triangle(start, ring_a[j], ring_a[i])
            self.triangle(end, ring_b[i], ring_b[j])


def add_accessors(doc: dict, binary: bytearray, geometry: Geometry) -> dict:
    if not geometry.vertices:
        raise ValueError("Empty seat mesh")
    accessors = []
    for values, kind in ((geometry.vertices, "POSITION"), (geometry.normals, "NORMAL")):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        for value in values:
            binary.extend(struct.pack("<3f", *value))
        view_index = len(doc["bufferViews"])
        doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(values) * 12, "target": 34962})
        accessor = {"bufferView": view_index, "componentType": 5126, "count": len(values), "type": "VEC3"}
        if kind == "POSITION":
            accessor["min"] = [min(v[i] for v in values) for i in range(3)]
            accessor["max"] = [max(v[i] for v in values) for i in range(3)]
        accessors.append(len(doc["accessors"]))
        doc["accessors"].append(accessor)
    return {"POSITION": accessors[0], "NORMAL": accessors[1]}


def make_seat(doc: dict, binary: bytearray, seat_name: str, anchor: tuple[float, float, float]) -> int:
    x, y, z0 = anchor
    # Four lightweight mesh groups keep the model easy to inspect and shade.
    groups = {name: Geometry() for name in ("upholstery", "shell", "frame", "harness")}
    soft, shell, frame, harness = groups["upholstery"], groups["shell"], groups["frame"], groups["harness"]

    # Low-profile seat pan and its upholstered top.
    shell.box((x + 0.055, y + 0.115, z0), (0.43, 0.07, 0.43))
    soft.box((x + 0.045, y + 0.158, z0), (0.40, 0.045, 0.40))
    # Reclined backrest: rearward tilt follows the stroller's inclined frame.
    shell.box((x + 0.155, y + 0.390, z0), (0.07, 0.47, 0.43), angle=-math.radians(18))
    soft.box((x + 0.112, y + 0.395, z0), (0.045, 0.43, 0.39), angle=-math.radians(18))

    # Side bolsters and arm rails bound the child seating area.
    for side in (-1, 1):
        zz = z0 + side * 0.198
        soft.box((x - 0.005, y + 0.225, zz), (0.32, 0.13, 0.045), angle=math.radians(7))
        frame.beam((x - 0.175, y + 0.255, zz), (x + 0.110, y + 0.285, zz), 0.012)
        frame.beam((x + 0.085, y + 0.275, zz), (x + 0.245, y + 0.520, zz), 0.014)
        frame.beam((x + 0.245, y + 0.520, zz), (x + 0.185, y + 0.595, zz), 0.014)

        # Broad molded cheek plates bring the seat shell down to the frame
        # attachment points. The blocky socket overlaps the chassis host so
        # the interface reads as a direct, load-bearing snap-in connection.
        host_z = side * 0.22
        shell.side_cheek(
            [(x - 0.075, y + 0.155), (x + 0.205, y + 0.145), (x + 0.080, y - 0.075)],
            side * 0.190,
            0.085,
        )
        shell.box((x + 0.015, y + 0.015, host_z), (0.185, 0.145, 0.105))
        # Large molded latch face and cross-pin make the insertion interface
        # legible from a 3/4 view without adding a slender support rod.
        shell.box((x + 0.015, y + 0.015, side * 0.252), (0.120, 0.095, 0.025))
        frame.beam((x + 0.015, y + 0.015, side * 0.247), (x + 0.015, y + 0.015, side * 0.285), 0.018)

    # Front restraint bar and an extended padded legrest. The broad shell
    # panel begins at the seat front and projects forward so it is visible as
    # a child leg support, rather than hiding as a tiny block beneath the pan.
    frame.beam((x - 0.155, y + 0.300, -0.188), (x - 0.155, y + 0.300, 0.188), 0.013)
    for side in (-1, 1):
        frame.beam((x - 0.155, y + 0.300, side * 0.188), (x - 0.105, y + 0.275, side * 0.198), 0.013)
    frame.beam((x - 0.120, y + 0.055, -0.13), (x - 0.120, y + 0.055, 0.13), 0.012)
    shell.box((x - 0.310, y + 0.087, 0.0), (0.32, 0.070, 0.38), angle=math.radians(8))
    soft.box((x - 0.310, y + 0.128, 0.0), (0.30, 0.035, 0.35), angle=math.radians(8))

    # Five-point harness strips and central buckle.
    for side in (-1, 1):
        harness.beam((x + 0.190, y + 0.535, side * 0.095), (x - 0.055, y + 0.205, side * 0.045), 0.009, sides=8)
        harness.beam((x - 0.055, y + 0.205, side * 0.045), (x - 0.105, y + 0.200, side * 0.018), 0.009, sides=8)
        harness.beam((x - 0.045, y + 0.205, side * 0.045), (x - 0.045, y + 0.205, 0.0), 0.009, sides=8)
    harness.box((x - 0.045, y + 0.198, 0.0), (0.035, 0.035, 0.045))

    material_indices = {}
    for name, color, metallic, roughness in (
        ("upholstery", [0.105, 0.145, 0.190, 1.0], 0.0, 0.82),
        ("shell", [0.075, 0.085, 0.095, 1.0], 0.12, 0.62),
        ("frame", [0.43, 0.49, 0.55, 1.0], 0.68, 0.34),
        ("harness", [0.72, 0.40, 0.13, 1.0], 0.0, 0.7),
    ):
        material_indices[name] = len(doc["materials"])
        doc["materials"].append({
            "name": f"D02_{name}",
            "pbrMetallicRoughness": {
                "baseColorFactor": color,
                "metallicFactor": metallic,
                "roughnessFactor": roughness,
            },
            "doubleSided": True,
        })

    primitives = []
    for name, geometry in groups.items():
        if not geometry.vertices:
            continue
        attributes = add_accessors(doc, binary, geometry)
        primitives.append({"attributes": attributes, "material": material_indices[name], "mode": 4})

    mesh_index = len(doc["meshes"])
    doc["meshes"].append({"name": seat_name, "primitives": primitives})
    node_index = len(doc["nodes"])
    doc["nodes"].append({"name": seat_name, "mesh": mesh_index})
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(node_index)
    return node_index


def main():
    doc, binary = read_glb(SOURCE)
    doc, binary = compact_active_scene(doc, binary)
    # The existing chassis contains low and high attachment hosts at these
    # measured centres: LOW x=.280/y=.450, HIGH x=.787/y=.830, z= +/- .220 m.
    make_seat(doc, binary, "D02_lower_S05_molded_side_mounts_concept", (0.280, 0.450, 0.0))
    make_seat(doc, binary, "D02_upper_S05_molded_side_mounts_concept", (0.787, 0.830, 0.0))
    write_glb(OUTPUT, doc, binary)
    print(f"Created {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
