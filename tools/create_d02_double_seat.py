#!/usr/bin/env python3
"""Build the six assembled D02 double-child-seat concept GLBs."""

from __future__ import annotations

import math
import pathlib
import struct

from generate_wheel_variants import write_glb
from prepare_d02_chassis import prepare_chassis, level_assembly


ROOT = pathlib.Path(__file__).resolve().parents[1]
SEAT_DROP = 0.085
STANDARD_MOUNT_Z = 0.220
RAIL_ANGLE = math.radians(37)


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
        """Extrude a clockwise convex side profile."""
        z0, z1 = z_center - thickness / 2, z_center + thickness / 2
        front = [(x, y, z0) for x, y in points]
        back = [(x, y, z1) for x, y in points]
        for i in range(1, len(points) - 1):
            self.triangle(front[0], front[i], front[i + 1])
            self.triangle(back[0], back[i + 1], back[i])
        for i in range(len(points)):
            j = (i + 1) % len(points)
            self.quad(front[j], front[i], back[i], back[j])

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

    def seat_pad(self, center, dims, corner, bevel, rear_rise):
        """Round the seat in plan and blend the back edge up into the backrest."""
        cx, cy, cz = center
        dx, dy, dz = (v / 2 for v in dims)

        def ring(level, inset):
            points = []
            for quadrant, (sx, sz) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
                for step in range(7):
                    angle = math.pi / 2 * (quadrant + step / 6)
                    x = sx * (dx - corner) + (corner - inset) * math.cos(angle)
                    z = sz * (dz - corner) + (corner - inset) * math.sin(angle)
                    ramp = max(0.0, min(1.0, (x - dx + 0.075) / 0.075))
                    rise = rear_rise * ramp**2 * (3 - 2 * ramp)
                    points.append((cx + x, cy + level + rise, cz + z))
            return points

        rings = [
            ring(-dy, bevel),
            ring(-dy + bevel, 0.0),
            ring(dy - bevel, 0.0),
            ring(dy, bevel),
        ]
        for lower, upper in zip(rings, rings[1:]):
            for i in range(len(lower)):
                j = (i + 1) % len(lower)
                self.quad(lower[i], upper[i], upper[j], lower[j])
        # Span across the paired edges so the two cap surfaces follow the
        # curved rear return, instead of running a diagonal through the rise.
        for boundary, upper in ((rings[0], False), (rings[-1], True)):
            end = len(boundary) - 1
            for i in range((end - 1) // 2):
                strip = (boundary[i], boundary[i + 1], boundary[end - i - 1], boundary[end - i])
                self.quad(*(reversed(strip) if upper else strip))

    def beam(self, start, end, radius, sides=12):
        axis = unit(sub(end, start))
        ref = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (0.0, 1.0, 0.0)
        self.tube([start, end], radius, ref, sides)

    def tube(self, path, radius, ref, sides=8):
        rings = []
        for i, point in enumerate(path):
            tangent = unit(sub(path[min(i + 1, len(path) - 1)], path[max(i - 1, 0)]))
            u = unit(cross(tangent, ref))
            v = unit(cross(tangent, u))
            ring = []
            for j in range(sides):
                angle = 2 * math.pi * j / sides
                ring.append(tuple(
                    point[k] + radius * (u[k] * math.cos(angle) + v[k] * math.sin(angle))
                    for k in range(3)
                ))
            rings.append(ring)
        for a, b in zip(rings, rings[1:]):
            for i in range(sides):
                j = (i + 1) % sides
                self.quad(a[i], a[j], b[j], b[i])
        for i in range(sides):
            j = (i + 1) % sides
            self.triangle(path[0], rings[0][j], rings[0][i])
            self.triangle(path[-1], rings[-1][i], rings[-1][j])


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


def add_canopy(fabric, trim, shell):
    # Each row sets the center and side coordinates of a transverse bow.
    # The edges converge toward the two pivots below the fabric.
    bows = [
        (0.278, 0.619, 0.236, 0.367, 0.224),
        (0.264, 0.688, 0.241, 0.404, 0.230),
        (0.136, 0.736, 0.188, 0.435, 0.236),
        (-0.016, 0.712, 0.107, 0.455, 0.234),
        (-0.149, 0.617, 0.015, 0.465, 0.226),
    ]

    def arch(bow):
        cx, cy, sx, sy, half_width = bow
        points = []
        for j in range(25):
            theta = math.pi * j / 24
            height = math.sin(theta) ** 0.52
            points.append((sx + (cx - sx) * height, sy + (cy - sy) * height, -half_width * math.cos(theta)))
        return points

    rows = [arch(bows[0])]
    for i, (left, right) in enumerate(zip(bows, bows[1:])):
        before = bows[max(i - 1, 0)]
        after = bows[min(i + 2, len(bows) - 1)]
        for j in range(1, 6):
            t = j / 5
            smooth = 3 * t**2 - 2 * t**3
            start_tangent = t * (1 - t)**2
            end_tangent = t**2 * (t - 1)
            rows.append(arch([
                x + (y - x) * smooth + (y - w) * start_tangent / 2 + (z - x) * end_tangent / 2
                for w, x, y, z in zip(before, left, right, after)
            ]))

    for back, front in zip(rows, rows[1:]):
        for j in range(len(back) - 1):
            fabric.quad(back[j], front[j], front[j + 1], back[j + 1])

    for row, radius in ((rows[0], 0.004), (rows[10], 0.003), (rows[-1], 0.007)):
        trim.tube(row, radius, (1.0, 0.0, 0.0))
    for side, edge in ((-1, 0), (1, -1)):
        hinge = (0.152, 0.330, side * 0.221)
        shell.beam((hinge[0], hinge[1], side * 0.200), (hinge[0], hinge[1], side * 0.237), 0.031)
        trim.tube([row[edge] for row in rows], 0.004, (0.0, 0.0, 1.0))
        for row in (rows[0], rows[-1]):
            trim.beam(hinge, row[edge], 0.006, sides=8)


def rail_point(u, v, z):
    cosine, sine = math.cos(RAIL_ANGLE), math.sin(RAIL_ANGLE)
    return u * cosine - v * sine, u * sine + v * cosine, z


def rail_box(geometry, u, v, z, dims):
    geometry.box(rail_point(u, v, z), dims, RAIL_ANGLE)


def add_socket(socket, latch, side, center_z):
    rail_box(socket, 0.003, 0.037, side * center_z, (0.112, 0.026, 0.046))
    # The center remains open from below. A shallow pad locates the insert.
    rail_box(socket, -0.010, 0.023, side * center_z, (0.028, 0.008, 0.022))
    for wall in (-1, 1):
        rail_box(socket, 0.003, 0.008, side * center_z + wall * 0.020, (0.112, 0.053, 0.006))
    latch.beam(
        rail_point(0.003, 0.009, side * (center_z + 0.021)),
        rail_point(0.003, 0.009, side * (center_z + 0.026)),
        0.014,
    )


def add_side_mount(panel, mount, latch, adapter, host_z):
    for side in (-1, 1):
        # One molded side panel is both the low sidewall and the socket support.
        panel.side_cheek(
            [(-0.110, 0.078), (-0.075, 0.127), (0.078, 0.145), (0.110, 0.117),
             (0.078, 0.061), (-0.030, 0.018), (-0.085, 0.026)],
            side * 0.191,
            0.020,
        )
        add_socket(mount, latch, side, STANDARD_MOUNT_Z)

        # Wide chassis: a tongue plugs into the standard socket from below.
        # Its offset bridge carries a second socket over the actual frame port.
        offset = host_z - STANDARD_MOUNT_Z
        if offset > 0.001:
            add_socket(adapter, latch, side, host_z)
            rail_box(adapter, -0.002, -0.004, side * STANDARD_MOUNT_Z, (0.060, 0.051, 0.026))
            rail_box(adapter, -0.002, -0.028, side * (STANDARD_MOUNT_Z + (offset - 0.016) / 2),
                     (0.062, 0.017, offset - 0.004))
            rail_box(adapter, -0.002, -0.015, side * (host_z - 0.020), (0.062, 0.013, 0.007))


def add_guardrail(rail, pivots):
    side_arm = [(0.081, 0.234, 0.196), (-0.035, 0.294, 0.202),
                (-0.169, 0.333, 0.202), (-0.201, 0.337, 0.191), (-0.215, 0.337, 0.168)]
    rail.tube(side_arm + [(x, y, -z) for x, y, z in reversed(side_arm)],
              0.014, (0.0, 1.0, 0.0), sides=12)
    for side in (-1, 1):
        x, y, _ = side_arm[0]
        pivots.beam((x, y, side * 0.179), (x, y, side * 0.208), 0.022)


def add_legrest(support, pad, rear):
    # Keep the rear tray under its own pan, behind the front seat envelope.
    # The front tray can project farther forward with a shallower slope.
    start = (-0.143, 0.109)
    toe = (-0.189, -0.086) if rear else (-0.323, -0.055)
    dx, dy = start[0] - toe[0], start[1] - toe[1]
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    cx, cy = (start[0] + toe[0]) / 2, (start[1] + toe[1]) / 2
    width = 0.328 if rear else 0.356
    support.box((cx, cy, 0.0), (length + 0.020, 0.033, width), angle)
    pad.box((cx - math.sin(angle) * 0.022, cy + math.cos(angle) * 0.022, 0.0),
            (length - 0.020, 0.015, width - 0.026), angle)
    support.box((toe[0] - 0.011, toe[1] - 0.004, 0.0), (0.061, 0.023, width))
    pad.box((toe[0] - 0.018, toe[1] + 0.011, 0.0), (0.046, 0.011, width - 0.018))


def make_seat(doc, binary, seat_name, anchor, host_z, rear=False):
    groups = {name: Geometry() for name in (
        "upholstery", "shell", "guardrail", "harness", "mount_cheek", "mounts", "mount_pin", "width_adapter",
        "legrest_shell", "legrest", "canopy", "canopy_trim",
    )}
    soft, shell, harness = (groups[name] for name in ("upholstery", "shell", "harness"))

    shell.seat_pad((-0.012, 0.126, 0.0), (0.294, 0.034, 0.394), 0.047, 0.007, 0.026)
    soft.seat_pad((-0.023, 0.159, 0.0), (0.256, 0.038, 0.374), 0.041, 0.010, 0.013)
    shell.box((0.151, 0.384, 0.0), (0.062, 0.445, 0.394), -math.radians(16))
    soft.box((0.111, 0.391, 0.0), (0.045, 0.425, 0.378), -math.radians(16))

    for side in (-1, 1):
        soft.box((0.009, 0.199, side * 0.177), (0.180, 0.037, 0.027), math.radians(7))
        shoulder = (0.128, 0.535, side * 0.097)
        lower_back = (0.066, 0.320, side * 0.068)
        buckle = (-0.063, 0.198, side * 0.012)
        harness.beam(shoulder, lower_back, 0.009, sides=8)
        harness.beam(lower_back, buckle, 0.009, sides=8)
        harness.beam((0.026, 0.195, side * 0.180), buckle, 0.009, sides=8)

    harness.box((-0.064, 0.196, 0.0), (0.032, 0.032, 0.044))
    harness.beam((-0.115, 0.180, 0.0), (-0.071, 0.200, 0.0), 0.010, sides=8)

    add_guardrail(groups["guardrail"], shell)
    add_legrest(groups["legrest_shell"], groups["legrest"], rear)
    add_canopy(groups["canopy"], groups["canopy_trim"], shell)
    for geometry in groups.values():
        geometry.vertices = [(x, y - SEAT_DROP, z) for x, y, z in geometry.vertices]
    add_side_mount(groups["mount_cheek"], groups["mounts"], groups["mount_pin"], groups["width_adapter"], host_z)

    material_indices = {}
    for name, color, metallic, roughness in (
        ("upholstery", [0.105, 0.145, 0.190, 1.0], 0.0, 0.82),
        ("shell", [0.075, 0.085, 0.095, 1.0], 0.12, 0.62),
        ("guardrail", [0.43, 0.49, 0.55, 1.0], 0.56, 0.42),
        ("harness", [0.72, 0.40, 0.13, 1.0], 0.0, 0.7),
        ("mount_cheek", [0.155, 0.258, 0.310, 1.0], 0.08, 0.62),
        ("mounts", [0.190, 0.415, 0.475, 1.0], 0.25, 0.40),
        ("mount_pin", [0.740, 0.804, 0.833, 1.0], 0.52, 0.34),
        ("width_adapter", [0.63, 0.32, 0.105, 1.0], 0.26, 0.46),
        ("legrest_shell", [0.076, 0.092, 0.107, 1.0], 0.08, 0.63),
        ("canopy", [0.115, 0.184, 0.235, 1.0], 0.0, 0.87),
        ("canopy_trim", [0.050, 0.077, 0.099, 1.0], 0.0, 0.76),
        ("legrest", [0.24, 0.32, 0.40, 1.0], 0.0, 0.82),
    ):
        if not groups[name].vertices:
            continue
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
        geometry.vertices = [
            (x + anchor[0], y + anchor[1], z)
            for x, y, z in geometry.vertices
        ]
        attributes = add_accessors(doc, binary, geometry)
        primitives.append({"attributes": attributes, "material": material_indices[name], "mode": 4})

    mesh_index = len(doc["meshes"])
    doc["meshes"].append({"name": seat_name, "primitives": primitives})
    node_index = len(doc["nodes"])
    doc["nodes"].append({"name": seat_name, "mesh": mesh_index})
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(node_index)
    return node_index


def port_position(doc, port):
    centers = []
    for side in (1, -1):
        prefix = f"narrow-open_host_{side}_{port}."
        node = next(node for node in doc["nodes"] if node["name"].startswith(prefix))
        primitive = doc["meshes"][node["mesh"]]["primitives"][0]
        accessor = doc["accessors"][primitive["attributes"]["POSITION"]]
        centers.append([(lo + hi) / 2 for lo, hi in zip(accessor["min"], accessor["max"])])
    front_back, height, _ = ((a + b) / 2 for a, b in zip(*centers))
    half_span = abs(centers[0][2] - centers[1][2]) / 2
    return (front_back, height), half_span


def main():
    output_dir = ROOT / "models/double_cabin"
    output_dir.mkdir(parents=True, exist_ok=True)
    for width in (550, 650):
        for wheel in ("S", "M", "L"):
            doc, binary = prepare_chassis(width, wheel)
            lower, lower_span = port_position(doc, "L")
            upper, upper_span = port_position(doc, "H")
            make_seat(doc, binary, "D02_Front", lower, lower_span)
            make_seat(doc, binary, "D02_Rear", upper, upper_span, rear=True)
            level_assembly(doc, binary)
            output = output_dir / f"D02_w{width}_state1_wheel_{wheel}.glb"
            write_glb(output, doc, binary)
            print(f"Created {output.relative_to(ROOT)} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
