#!/usr/bin/env python3
"""Build the six assembled D02 double-child-seat concept GLBs."""

from __future__ import annotations

import math
import pathlib
import struct

from generate_wheel_variants import read_positions, write_glb
from prepare_d02_chassis import prepare_chassis, level_assembly


ROOT = pathlib.Path(__file__).resolve().parents[1]
SEAT_DROP = 0.085
FRONT_RECLINE_DROP = 0.065
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


def accessor(doc, binary, values, kind, *, vertex=False, bounds=False):
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    width = 1 if kind == "SCALAR" else int(kind[-1])
    pack = struct.Struct("<" + "f" * width).pack
    for value in values:
        binary.extend(pack(*value))
    view = {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
    if vertex:
        view["target"] = 34962
    access = {"bufferView": len(doc["bufferViews"]), "componentType": 5126, "count": len(values), "type": kind}
    if bounds:
        access["min"] = [min(axis) for axis in zip(*values)]
        access["max"] = [max(axis) for axis in zip(*values)]
    doc["bufferViews"].append(view)
    doc["accessors"].append(access)
    return len(doc["accessors"]) - 1


def rotation(axis, angle):
    half_angle = math.radians(angle) / 2
    return [*(value * math.sin(half_angle) for value in axis), math.cos(half_angle)]


def clip(doc, binary, name, tracks):
    samplers, channels = [], []
    for node, path, times, values, *interpolation in tracks:
        samplers.append({
            "input": accessor(doc, binary, [(t,) for t in times], "SCALAR", bounds=True),
            "output": accessor(doc, binary, values, {"rotation": "VEC4", "translation": "VEC3", "scale": "VEC3", "weights": "SCALAR"}[path]),
            "interpolation": interpolation[0] if interpolation else "LINEAR",
        })
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})
    doc.setdefault("animations", []).append({"name": name, "samplers": samplers, "channels": channels})


def canopy_geometry(opening, lift=0.0):
    # Each row sets the center and side coordinates of a transverse bow.
    # The edges converge toward the two pivots below the fabric.
    extended = [
        (0.278, 0.619, 0.236, 0.367, 0.224),
        (0.264, 0.688, 0.241, 0.404, 0.230),
        (0.136, 0.736, 0.188, 0.435, 0.236),
        (-0.016, 0.712, 0.107, 0.455, 0.234),
        (-0.149, 0.617, 0.015, 0.465, 0.226),
    ]
    folded = [
        (0.278, 0.619, 0.236, 0.367, 0.224),
        (0.279, 0.632, 0.237, 0.373, 0.225),
        (0.280, 0.644, 0.238, 0.379, 0.226),
        (0.281, 0.656, 0.239, 0.385, 0.227),
        (0.282, 0.668, 0.240, 0.391, 0.228),
    ]
    bows = []
    for i, (closed, opened) in enumerate(zip(folded, extended)):
        cx, cy, sx, sy, width = [a + opening * (b - a) for a, b in zip(closed, opened)]
        # Intermediate bows swing above the backrest before extending forward.
        rise = lift * i / (len(extended) - 1)
        # Keep the arches above the sweep of the bottom-hinged backrest.
        # The lower side attachment points stay in their original sockets.
        bows.append((cx, cy + rise + 0.067, sx, sy + rise * 1.15 + 0.067, width))
    fabric, trim = Geometry(), Geometry()

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
        trim.tube([row[edge] for row in rows], 0.004, (0.0, 0.0, 1.0))
        for row in (rows[0], rows[-1]):
            trim.beam(hinge, row[edge], 0.006, sides=8)
    return {"canopy": fabric, "canopy_trim": trim}


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


def add_side_mount(panel, mount, latch, adapter, host_z, panel_drop=0.0):
    for side in (-1, 1):
        # One molded side panel is both the low sidewall and the socket support.
        profile = [(-0.174, 0.055), (-0.174, 0.108), (0.078, 0.145), (0.110, 0.117),
                   (0.078, 0.061), (-0.030, 0.018), (-0.125, 0.026)]
        panel.side_cheek(
            [(x, y - panel_drop) for x, y in profile],
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


def guard_cradle(geometry, center):
    """Low, open-top dock. The free terminal leaves it upward."""
    x, y, z = center
    arcs = []
    for end in (x - 0.007, x + 0.007):
        rings = []
        for radius in (0.023, 0.016):
            rings.append([
                (end, y - radius * math.sin(math.pi * i / 12), z + radius * math.cos(math.pi * i / 12))
                for i in range(13)
            ])
        arcs.append(rings)
    (outer_start, inner_start), (outer_end, inner_end) = arcs
    for i in range(12):
        j = i + 1
        geometry.quad(outer_start[i], outer_end[i], outer_end[j], outer_start[j])
        geometry.quad(inner_start[j], inner_end[j], inner_end[i], inner_start[i])
        geometry.quad(outer_start[j], inner_start[j], inner_start[i], outer_start[i])
        geometry.quad(outer_end[i], inner_end[i], inner_end[j], outer_end[j])
    geometry.quad(outer_start[0], inner_start[0], inner_end[0], outer_end[0])
    geometry.quad(outer_end[-1], inner_end[-1], inner_start[-1], outer_start[-1])


def add_guardrail(doc, binary, name, base):
    side_arm = [(-0.180, 0.219, 0.199), (-0.193, 0.262, 0.211), (-0.205, 0.315, 0.203),
                (-0.211, 0.334, 0.191), (-0.215, 0.337, 0.168)]
    x, y, z = side_arm[0]
    left, right = (x, y, z), (x, y, -z)
    rail = Geometry()
    rail.tube(side_arm + [(a, b, -c) for a, b, c in reversed(side_arm)],
              0.014, (1.0, 0.0, 0.0), sides=12)
    for end in (left, right):
        _, _, lateral = end
        side = math.copysign(1, lateral)
        dock_x = x + 0.024
        # Only the axial pin enters the dock. The rail bends forward of it,
        # leaving both the retained dock and the release button out of its sweep.
        rail.beam((x - 0.008, y, lateral), (dock_x + 0.004, y, lateral), 0.012)
        guard_cradle(base["guard_dock"], (dock_x, y, lateral))
        base["guard_dock"].box((dock_x, y - 0.025, side * 0.192), (0.014, 0.018, 0.024))
        base["guard_release"].beam((dock_x, y - 0.010, lateral + side * 0.017),
                                   (dock_x, y - 0.010, lateral + side * 0.024), 0.008)

    # Each endpoint is a separate pivot. At most one is opened at a time.
    bar = mesh_node(doc, binary, name + "_FullGuard", {"guardrail": rail}, origin=left)
    left_hinge = add_node(doc, name=name + "_LeftGuardPivot", translation=sub(left, right), children=[bar])
    right_hinge = add_node(doc, name=name + "_RightGuardPivot", translation=right, children=[left_hinge])
    removal = add_node(doc, name=name + "_GuardRemoval", children=[right_hinge])
    for release_side, hinge, angle in (("Left", right_hinge, -110), ("Right", left_hinge, 110)):
        clip(doc, binary, name + "_guard" + release_side,
             [(hinge, "rotation", (0, 1), [rotation(AXIS_X, a) for a in (0, angle)])])
    clip(doc, binary, name + "_guardRemoved", [
        (removal, "translation", (0, 0.28, 0.93, 1),
         [(0, 0, 0), (0, 0.055, 0), (0, 0.055, 0.505), (0, 0.055, 0.505)]),
        (bar, "scale", (0, 0.94, 1), [(1, 1, 1), (0, 0, 0), (0, 0, 0)], "STEP"),
    ])
    return removal


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
    pad_trim = 0.010
    support.box((cx, cy, 0.0), (length + 0.020, 0.033, width), angle)
    pad.box((cx - math.sin(angle) * 0.022 - math.cos(angle) * pad_trim / 2,
             cy + math.cos(angle) * 0.022 - math.sin(angle) * pad_trim / 2, 0.0),
            (length - 0.020 - pad_trim, 0.015, width - 0.026), angle)
    support.box((toe[0] - 0.011, toe[1] - 0.004, 0.0), (0.061, 0.023, width))
    pad.box((toe[0] - 0.018, toe[1] + 0.011, 0.0), (0.046, 0.011, width - 0.018))


MATERIALS = {
    "upholstery": ([0.105, 0.145, 0.190, 1.0], 0.0, 0.82),
    "shell": ([0.075, 0.085, 0.095, 1.0], 0.12, 0.62),
    "guardrail": ([0.43, 0.49, 0.55, 1.0], 0.56, 0.42),
    "guard_dock": ([0.098, 0.171, 0.205, 1.0], 0.15, 0.55),
    "guard_release": ([0.85, 0.45, 0.13, 1.0], 0.08, 0.48),
    "harness": ([0.72, 0.40, 0.13, 1.0], 0.0, 0.7),
    "mount_cheek": ([0.155, 0.258, 0.310, 1.0], 0.08, 0.62),
    "mounts": ([0.190, 0.415, 0.475, 1.0], 0.25, 0.40),
    "mount_pin": ([0.740, 0.804, 0.833, 1.0], 0.52, 0.34),
    "width_adapter": ([0.63, 0.32, 0.105, 1.0], 0.26, 0.46),
    "legrest_shell": ([0.076, 0.092, 0.107, 1.0], 0.08, 0.63),
    "legrest": ([0.24, 0.32, 0.40, 1.0], 0.0, 0.82),
    "canopy": ([0.115, 0.184, 0.235, 1.0], 0.0, 0.87),
    "canopy_trim": ([0.050, 0.077, 0.099, 1.0], 0.0, 0.76),
    "canopy_stay": ([0.112, 0.177, 0.212, 1.0], 0.10, 0.60),
}
AXIS_X = (1.0, 0.0, 0.0)
AXIS_Z = (0.0, 0.0, 1.0)
BACK_PIVOT = (0.091, 0.163, 0.0)
BACK_DEGREES = (95, 115, 135, 155, 175)
BACK_ROTATIONS = tuple(106 - degrees for degrees in BACK_DEGREES)
BACK_TIMES = (0, 0.25, 0.5, 0.75, 1)
THREE_STEPS = (0.0, 0.5, 1.0)
CANOPY_EYE = (0.152, 0.330, 0.0)


def add_node(doc, **node):
    index = len(doc["nodes"])
    doc["nodes"].append(node)
    return index


def mesh_node(doc, binary, name, groups, *, origin=(0.0, 0.0, 0.0), morphs=(), **node):
    primitives = []
    existing = {m.get("name"): i for i, m in enumerate(doc["materials"])}
    for material, geometry in groups.items():
        if not geometry.vertices:
            continue
        material_name = "D02_" + material
        if material_name not in existing:
            color, metallic, roughness = MATERIALS[material]
            existing[material_name] = len(doc["materials"])
            doc["materials"].append({
                "name": material_name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": color,
                    "metallicFactor": metallic,
                    "roughnessFactor": roughness,
                },
                "doubleSided": True,
            })
        attributes = {
            "POSITION": accessor(doc, binary, [sub(v, origin) for v in geometry.vertices], "VEC3", vertex=True, bounds=True),
            "NORMAL": accessor(doc, binary, geometry.normals, "VEC3", vertex=True),
        }
        primitive = {"attributes": attributes, "material": existing[material_name], "mode": 4}
        if morphs:
            primitive["targets"] = []
            for target in morphs:
                if len(target[material].vertices) != len(geometry.vertices):
                    raise ValueError(f"Mismatched topology in {name} morph")
                primitive["targets"].append({
                    "POSITION": accessor(doc, binary, [sub(a,b) for a,b in zip(target[material].vertices, geometry.vertices)],
                                         "VEC3", vertex=True, bounds=True),
                    "NORMAL": accessor(doc, binary, [sub(a,b) for a,b in zip(target[material].normals, geometry.normals)],
                                       "VEC3", vertex=True),
                })
        primitives.append(primitive)
    mesh = {"name": name, "primitives": primitives}
    if morphs:
        mesh["weights"] = [0.0] * len(morphs)
    mesh_id = len(doc["meshes"])
    doc["meshes"].append(mesh)
    return add_node(doc, name=name, mesh=mesh_id, **node)


def shoulder_straps(angle):
    harness = Geometry()
    sine, cosine = math.sin(math.radians(angle)), math.cos(math.radians(angle))

    def follow_back(point):
        x, y, z = sub(point, BACK_PIVOT)
        return (x * cosine - y * sine + BACK_PIVOT[0],
                x * sine + y * cosine + BACK_PIVOT[1], z)

    for side in (-1, 1):
        shoulder = follow_back((0.128, 0.535, side * 0.097))
        lower = follow_back((0.066, 0.320, side * 0.068))
        harness.beam(shoulder, lower, 0.009, sides=8)
        harness.beam(lower, (-0.063, 0.198, side * 0.012), 0.009, sides=8)
    return {"harness": harness}


def canopy_link_geometry(eye):
    x, y, _ = eye
    dx, dy = x - CANOPY_EYE[0], y - CANOPY_EYE[1]
    stay = Geometry()
    for side in (-1, 1):
        stay.tube([(a, b, side * z) for a, b, z in (
            (0.006, 0.171, 0.191), (0.006, 0.213, 0.191), (0.029, 0.240, 0.215),
            (0.087 + dx / 2, 0.270 + dy / 2, 0.220), (x, y, 0.221),
        )], 0.010, AXIS_Z)
    return {"canopy_stay": stay}


def add_linked_canopy(doc, binary, name, rear):
    # The link has different travel on the two seats: the front canopy must
    # stay out of the adjacent seat's guardrail and legrest envelope.
    pivot = CANOPY_EYE if rear else (0.152, 0.780, 0.0)
    travel = (0.025, -0.015, 0.0) if rear else (0.008, 0.0, 0.0)
    angle = -18 if rear else -20
    canopy = mesh_node(doc, binary, name + "_Canopy", canopy_geometry(1.0), origin=pivot, morphs=[
        canopy_geometry(0), canopy_geometry(1 / 3, 0.115), canopy_geometry(2 / 3, 0.060)
    ])
    weights = [(w,) for w in (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)]
    clip(doc, binary, name + "_canopy", [(canopy, "weights", (0, 1 / 3, 2 / 3, 1), weights)])

    hinges = Geometry()
    x, y, _ = CANOPY_EYE
    for side in (-1, 1):
        hinges.beam((x, y, side * 0.205), (x, y, side * 0.237), 0.031)
    hinge_node = mesh_node(doc, binary, name + "_CanopyHinges", {"canopy_stay": hinges}, origin=pivot)
    carrier = add_node(doc, name=name + "_CanopyCarrier", translation=pivot, children=[canopy, hinge_node])

    eyes, positions, rotations = [], [], []
    hx, hy, _ = sub(CANOPY_EYE, pivot)
    for t in BACK_TIMES:
        a = math.radians(angle * t)
        c, s = math.cos(a), math.sin(a)
        position = tuple(p + d * t for p, d in zip(pivot, travel))
        eyes.append((position[0] + c * hx - s * hy, position[1] + s * hx + c * hy, 0.0))
        positions.append(position)
        rotations.append(rotation(AXIS_Z, angle * t))
    link = mesh_node(doc, binary, name + "_CanopyStay", canopy_link_geometry(CANOPY_EYE),
                     morphs=[canopy_link_geometry(eye) for eye in eyes[1:]])
    link_weights = [(1.0 if key == target else 0.0,) for key in range(5) for target in range(1, 5)]
    return (carrier, link), [
        (carrier, "rotation", BACK_TIMES, rotations),
        (carrier, "translation", BACK_TIMES, positions),
        (link, "weights", BACK_TIMES, link_weights),
    ]


def handle_extension(doc, binary, level):
    """Telescoping handle motion keyed ahead of the deep rear recline."""
    angle, _ = level
    c, s = math.cos(angle), math.sin(angle)
    reach = (c * 0.200 - s * 0.154, s * 0.200 + c * 0.154, 0.0)
    forward = unit(reach)
    times = (0, 0.16, 0.43, 1)
    tracks = []
    for index, node in enumerate(doc["nodes"]):
        label = node.get("name", "")
        if label.startswith(("round_handle.", "ground_pad_")):
            tracks.append((index, "translation", times, [(0.0, 0.0, 0.0)] * 2 + [reach] * 2))
        elif label.startswith("handle_slider_"):
            mesh = doc["meshes"][node["mesh"]]
            mesh["weights"] = [0.0]
            for primitive in mesh["primitives"]:
                positions = read_positions(doc, binary, primitive["attributes"]["POSITION"])
                projection = [sum(v[i] * forward[i] for i in range(3)) for v in positions]
                lo, hi = min(projection), max(projection)
                steps = [(p - lo) / (hi - lo) for p in projection]
                offsets = [tuple(r * t for r in reach) for t in steps]
                primitive["targets"] = [{"POSITION": accessor(doc, binary, offsets, "VEC3", vertex=True, bounds=True)}]
            tracks.append((index, "weights", times, [(0.0,), (0.0,), (1.0,), (1.0,)]))
    return tracks


def make_seat(doc, binary, name, anchor, host_z, level, rear=False):
    base = {key: Geometry() for key in ("upholstery", "shell", "guard_dock", "harness", "guard_release")}
    soft, shell, harness = (base[key] for key in ("upholstery", "shell", "harness"))
    soft.seat_pad((-0.023, 0.159, 0.0), (0.256, 0.038, 0.374), 0.041, 0.010, 0.013)
    shell.seat_pad((-0.012, 0.126, 0.0), (0.294, 0.034, 0.394), 0.047, 0.007, 0.026)
    for side in (-1, 1):
        soft.box((0.009, 0.199, side * 0.177), (0.180, 0.037, 0.027), math.radians(7))
        harness.beam((0.026, 0.195, side * 0.180), (-0.063, 0.198, side * 0.012), 0.009, sides=8)
    harness.box((-0.064, 0.196, 0.0), (0.032, 0.032, 0.044))
    harness.beam((-0.115, 0.180, 0.0), (-0.071, 0.200, 0.0), 0.010, sides=8)

    bar = add_guardrail(doc, binary, name, base)

    canopy_nodes, canopy_tracks = add_linked_canopy(doc, binary, name, rear)

    legs = {key: Geometry() for key in ("legrest_shell", "legrest")}
    add_legrest(legs["legrest_shell"], legs["legrest"], rear)
    leg_hinge = (-0.143, 0.109, 0.0)
    legrest = mesh_node(doc, binary, name + "_Legrest", legs, origin=leg_hinge, translation=leg_hinge)
    # The rear legrest has a fourth, under-seat position for front recline.
    leg_angles = (0, -10, -20, 93) if rear else (0, -16, -32)
    leg_times = (0, 0.25, 0.5, 1) if rear else THREE_STEPS
    clip(doc, binary, name + "_legrest", [(legrest, "rotation", leg_times, [rotation(AXIS_Z, a) for a in leg_angles])])

    moving_back = {key: Geometry() for key in ("shell", "upholstery")}
    moving_back["shell"].box((0.151, 0.384, 0.0), (0.062, 0.445, 0.394), -math.radians(16))
    moving_back["upholstery"].box((0.111, 0.391, 0.0), (0.045, 0.425, 0.378), -math.radians(16))
    backrest = mesh_node(doc, binary, name + "_Backrest", moving_back, origin=BACK_PIVOT, translation=BACK_PIVOT,
                         rotation=rotation(AXIS_Z, BACK_ROTATIONS[0]))
    straps = mesh_node(doc, binary, name + "_ShoulderStraps", shoulder_straps(BACK_ROTATIONS[0]),
                       morphs=[shoulder_straps(a) for a in BACK_ROTATIONS[1:]])
    strap_weights = [(1.0 if key == target else 0.0,) for key in range(5) for target in range(1, 5)]
    back_tracks = [
        (backrest, "rotation", BACK_TIMES, [rotation(AXIS_Z, a) for a in BACK_ROTATIONS]),
        (straps, "weights", BACK_TIMES, strap_weights),
    ]
    back_tracks.extend(canopy_tracks)
    if rear:
        back_tracks.extend(handle_extension(doc, binary, level))
    clip(doc, binary, name + "_backrest", back_tracks)

    panel_drop = 0.0 if rear else FRONT_RECLINE_DROP
    seat = mesh_node(doc, binary, name + "_Shell", base, translation=(0.0, -SEAT_DROP - panel_drop, 0.0),
                     children=[bar, *canopy_nodes, legrest, backrest, straps])
    mounts = {key: Geometry() for key in ("mount_cheek", "mounts", "mount_pin", "width_adapter")}
    add_side_mount(mounts["mount_cheek"], mounts["mounts"], mounts["mount_pin"], mounts["width_adapter"], host_z, panel_drop)
    angle, ground = level
    c, s = math.cos(angle), math.sin(angle)
    x, y = anchor
    root = mesh_node(doc, binary, name, mounts, children=[seat],
                     translation=(c*x - s*y, s*x + c*y - ground, 0.0),
                     rotation=rotation(AXIS_Z, math.degrees(angle)))
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(root)


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
            level = level_assembly(doc, binary)
            make_seat(doc, binary, "D02_Front", lower, lower_span, level)
            make_seat(doc, binary, "D02_Rear", upper, upper_span, level, rear=True)
            output = output_dir / f"D02_w{width}_state1_wheel_{wheel}.glb"
            write_glb(output, doc, binary)
            print(f"Created {output.relative_to(ROOT)} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
