#!/usr/bin/env python3
"""Build the six assembled D02 double-child-seat concept GLBs."""

from __future__ import annotations

import math
import pathlib
import struct

from d02_front_mount import relocate_front_ports, relocate_high_ports
from generate_wheel_variants import read_positions, write_glb
from prepare_d02_chassis import prepare_chassis, level_assembly
from universal_mount import (
    FRAME_LAND_Y, INSERT_DEPTH, INSERT_X, INSERT_Z, LOCK_CLEARANCE, LOCK_Y,
    RECEIVER_OUTER_X, RECEIVER_OUTER_Z, receiver_geometry, standardize_receivers,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SEAT_SHELL_DROP = 0.150
FRONT_PORT_ADVANCE = 0.150
FRONT_PORT_DROP = 0.080
HIGH_PORT_SETBACK = 0.060
HIGH_PORT_RISE = 0.005
STANDARD_MOUNT_Z = 0.256
# Taller station supports clear the 650 offset pocket above the upper rail.
D02_RISE = 0.030
D02_LAND_Y = FRAME_LAND_Y + D02_RISE
D02_LOCK_Y = LOCK_Y + D02_RISE


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

    def seat_pad(self, center, dims, corner, bevel, rear_rise, *, front_width=None):
        """Round the seat in plan and blend the back edge up into the backrest."""
        cx, cy, cz = center
        dx, dy, dz = (v / 2 for v in dims)
        taper = 0.0 if front_width is None else 1.0 - front_width / dims[2]

        def ring(level, inset):
            points = []
            for quadrant, (sx, sz) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
                for step in range(7):
                    angle = math.pi / 2 * (quadrant + step / 6)
                    x = sx * (dx - corner) + (corner - inset) * math.cos(angle)
                    z = sz * (dz - corner) + (corner - inset) * math.sin(angle)
                    ramp = max(0.0, min(1.0, (x - dx + 0.075) / 0.075))
                    rise = rear_rise * ramp**2 * (3 - 2 * ramp)
                    width_scale = 1.0 - taper * (dx - x) / (2 * dx)
                    points.append((cx + x, cy + level + rise, cz + z * width_scale))
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


# Center and side coordinates of the transverse bows, from the fixed rear
# seam to the moving leading bow. Each row lists center X/Y, side X/Y, and
# half width. The leading bow packs behind the hinge and fans forward about 170°.
CANOPY_REAR_BOW = (0.266, 0.637, 0.256, 0.581, 0.188)
CANOPY_BOW_STOPS = (
    (
        (0.268, 0.623, 0.256, 0.577, 0.188), (0.271, 0.606, 0.257, 0.573, 0.189),
        (0.272, 0.587, 0.258, 0.566, 0.190), (0.270, 0.564, 0.258, 0.556, 0.191),
        (0.269, 0.541, 0.258, 0.545, 0.191),
    ), (
        (0.269, 0.644, 0.256, 0.584, 0.189), (0.269, 0.651, 0.256, 0.590, 0.190),
        (0.265, 0.657, 0.254, 0.597, 0.191), (0.258, 0.661, 0.249, 0.605, 0.192),
        (0.250, 0.663, 0.244, 0.613, 0.191),
    ), (
        (0.253, 0.657, 0.247, 0.595, 0.188), (0.231, 0.679, 0.230, 0.610, 0.190),
        (0.203, 0.699, 0.211, 0.624, 0.191), (0.170, 0.702, 0.185, 0.634, 0.192),
        (0.133, 0.711, 0.160, 0.638, 0.191),
    ), (
        (0.247, 0.653, 0.251, 0.599, 0.189), (0.221, 0.683, 0.233, 0.631, 0.191),
        (0.119, 0.714, 0.151, 0.629, 0.192), (0.029, 0.712, 0.110, 0.630, 0.192),
        (-0.044, 0.700, 0.083, 0.647, 0.190),
    ), (
        (0.253, 0.654, 0.250, 0.602, 0.190), (0.225, 0.688, 0.234, 0.635, 0.190),
        (0.112, 0.719, 0.144, 0.624, 0.193), (-0.017, 0.709, 0.088, 0.622, 0.193),
        (-0.099, 0.688, 0.065, 0.658, 0.189),
    ),
)


def canopy_geometry(stop):
    bows = [CANOPY_REAR_BOW, *CANOPY_BOW_STOPS[stop]]
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

    for row, radius in ((rows[0], 0.004), (rows[10], 0.003), (rows[20], 0.003), (rows[-1], 0.007)):
        trim.tube(row, radius, (1.0, 0.0, 0.0))
    for side, edge in ((-1, 0), (1, -1)):
        hinge = (*CANOPY_EYE, side * 0.1997)
        trim.tube([row[edge] for row in rows], 0.004, (0.0, 0.0, 1.0))
        elbow = (0.251, 0.605, side * 0.1997)
        trim.tube([hinge, elbow, rows[0][edge]], 0.0025, AXIS_Z)
        end = rows[-1][edge]
        forward = smoothstep((0.240 - end[0]) / 0.105)
        guide = (0.256 - 0.121 * forward, 0.598 + 0.027 * forward, side * 0.1997)
        trim.tube([hinge, elbow, guide, end], 0.0025, AXIS_Z)
    return {"canopy": fabric, "canopy_trim": trim}


def insert_shoe(shoe, latch, side, span):
    """The two longitudinal halves and the centered lock fit in either direction."""
    z = side * span
    stem = INSERT_X / 2 - LOCK_CLEARANCE
    for end in (-1, 1):
        shoe.box((end * (LOCK_CLEARANCE + stem / 2), D02_LAND_Y - INSERT_DEPTH / 2, z),
                 (stem, INSERT_DEPTH, INSERT_Z))
    for lo, hi in ((D02_LAND_Y - INSERT_DEPTH, D02_LOCK_Y - LOCK_CLEARANCE),
                   (D02_LOCK_Y + LOCK_CLEARANCE, D02_LAND_Y)):
        shoe.box((0, (lo + hi) / 2, z), (2 * LOCK_CLEARANCE, hi - lo, INSERT_Z))
    latch.beam((0, D02_LOCK_Y, z), (0, D02_LOCK_Y, z + side * (RECEIVER_OUTER_Z / 2 + 0.002)),
               0.0038, sides=16)


def add_side_mount(panel, mount, shoe, latch):
    for side in (-1, 1):
        # The directional side shell and the symmetric latch pod are molded together.
        profile = [(-0.174, -0.010), (-0.174, 0.043), (0.078, 0.080), (0.110, 0.052),
                   (0.078, -0.004), (-0.030, -0.047), (-0.125, -0.039)]
        panel.side_cheek(profile, side * 0.191, 0.020)
        panel.side_cheek([(-0.048, 0.045), (-0.035, D02_LAND_Y + 0.020),
                          (0.035, D02_LAND_Y + 0.020), (0.048, 0.045)], side * 0.191, 0.021)
        inside, outside = 0.193, STANDARD_MOUNT_Z + RECEIVER_OUTER_Z / 2 - 0.005
        mount.box((0, D02_LAND_Y + 0.012, side * (inside + outside) / 2),
                  (RECEIVER_OUTER_X, 0.024, outside - inside))
        insert_shoe(shoe, latch, side, STANDARD_MOUNT_Z)
        latch.beam((0, D02_LAND_Y + 0.012, side * (outside - 0.002)),
                   (0, D02_LAND_Y + 0.012, side * (outside + 0.004)), 0.007)


def add_width_adapter(doc, binary, station, centers):
    """The optional 650 spacer mates to the frame and carries the standard pocket."""
    for port_side in (-1, 1):
        x, y, z = centers[station, port_side]
        side = math.copysign(1, z)
        adapter, latch = Geometry(), Geometry()
        span = abs(z)
        insert_shoe(adapter, latch, side, span)
        walls, normals = receiver_geometry(include_support=False)
        adapter.vertices.extend((a, b + D02_RISE, c + side * STANDARD_MOUNT_Z) for a, b, c in walls)
        adapter.normals.extend(normals)

        # Bridge around the centered pins, leaving each lock accessible.
        inner_end = STANDARD_MOUNT_Z + RECEIVER_OUTER_Z / 2 - 0.004
        outer_end = span + INSERT_Z / 2
        for end in (-1, 1):
            adapter.box((end * 0.026, D02_LAND_Y + 0.006, side * (inner_end + outer_end) / 2),
                        (0.012, 0.012, outer_end - inner_end))
        adapter.box((0, D02_LAND_Y + 0.006, z), (RECEIVER_OUTER_X, 0.012, RECEIVER_OUTER_Z))
        for geometry in (adapter, latch):
            geometry.vertices = [(a + x, b + y, c) for a, b, c in geometry.vertices]
        index = mesh_node(doc, binary, f"D02_WideAdapter_{station}_{port_side}", {"width_adapter": adapter, "mount_pin": latch})
        doc["scenes"][doc.get("scene", 0)]["nodes"].append(index)


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
    side_arm = [(-0.180, 0.219, 0.174), (-0.193, 0.262, 0.181), (-0.205, 0.315, 0.185),
                (-0.211, 0.334, 0.177), (-0.215, 0.337, 0.163)]
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
        base["guard_dock"].box((dock_x, y - 0.025, side * 0.174), (0.014, 0.018, 0.024))
        base["guard_release"].beam((dock_x, y - 0.010, lateral + side * 0.013),
                                   (dock_x, y - 0.010, lateral + side * 0.018), 0.007)

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


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def textile_strip(sections):
    """Make a thin, hemmed strip between rows in the seat's side plane."""
    cloth = Geometry()
    top, bottom = [], []
    for x, y, half_width, nx, ny in sections:
        top.append(((x, y, -half_width), (x, y, half_width)))
        bottom.append(((x - 0.005 * nx, y - 0.005 * ny, -half_width),
                       (x - 0.005 * nx, y - 0.005 * ny, half_width)))
    for a, b, c, d in zip(top, top[1:], bottom, bottom[1:]):
        cloth.quad(a[0], b[0], b[1], a[1])
        cloth.quad(c[1], d[1], d[0], c[0])
        cloth.quad(a[0], c[0], d[0], b[0])
        cloth.quad(b[1], d[1], c[1], a[1])
    cloth.quad(*top[0], *reversed(bottom[0]))
    cloth.quad(*bottom[-1], *reversed(top[-1]))
    return {"upholstery": cloth}


def textile_motion(doc, binary, name, shape, positions):
    # Subdivide the cover path so a seam follows the circular hinge motion
    # between discrete stops, too. The rigid part keeps its own rotation track.
    subdivisions = 8
    angles = [a + (b - a) * step / subdivisions
              for a, b in zip(positions, positions[1:]) for step in range(subdivisions)] + [positions[-1]]
    frames = [shape(angle) for angle in angles]
    node = mesh_node(doc, binary, name, frames[0], morphs=frames[1:])
    steps = len(frames) - 1
    weights = [(float(i == j),) for i in range(steps + 1) for j in range(1, steps + 1)]
    return node, (node, "weights", tuple(i / steps for i in range(steps + 1)), weights)


def knee_cover(angle, half_width):
    # The visible fabric feeds out from beneath the front cushion lip. Its
    # outer wrap goes around the front of the axle throughout the full fold.
    x, y, _ = LEG_AXIS
    seat = (-0.138 - x, 0.180 - y)
    deck = (-0.045, 0.068)
    start = math.atan2(seat[1], seat[0])
    end = math.atan2(deck[1], deck[0]) + math.radians(angle)
    start_radius, end_radius = math.hypot(*seat), math.hypot(*deck)
    fold = smoothstep((angle - 110) / 70)
    pleat = smoothstep((angle - 136) / 34)
    wrap_radius = 0.076 - fold * 0.041
    sections = []
    for i in range(17):
        t = i / 16
        theta = start + (end - start) * t
        c, s = math.cos(theta), math.sin(theta)
        r = wrap_radius + (start_radius - wrap_radius) * smoothstep(1 - t * 3)
        r += (end_radius - wrap_radius) * smoothstep(t * 3 - 2)
        width = 0.164 + (half_width - 0.164) * smoothstep(t)
        sides = smoothstep((t - 0.37) / 0.18) * smoothstep((0.96 - t) / 0.12)
        width -= 0.043 * pleat * sides
        # The lower portion tucks rearward into the fold as the footrest inverts.
        tuck = 0.033 * fold * math.sin(math.pi * max(0.0, 2 * t - 1))
        sections.append((x + r * c + tuck, y + r * s, width, c, s))
    return textile_strip(sections)


def add_legrest(doc, binary, name, rear):
    # The offset arms bring the pad up to the seat lip when deployed and
    # carry it below the chassis cross tube when inverted under the seat.
    axis = LEG_AXIS
    length, width = (0.152, 0.292) if rear else (0.173, 0.304)
    center_x = -0.018 - length / 2
    tray, pad, arms, brackets = Geometry(), Geometry(), Geometry(), Geometry()
    # The tapered toe clears the front tires as the casters turn beneath it.
    # Recess the hard shell at the hinge end to leave a pocket for the fold.
    hinge_inset = 0.022
    shell_width = width - (width - 0.214) * hinge_inset / length
    tray.seat_pad((center_x - hinge_inset / 2, 0.050, 0), (length - hinge_inset, 0.018, shell_width),
                  0.022, 0.004, 0, front_width=0.214)
    pad.seat_pad((center_x, 0.060, 0), (length - 0.012, 0.013, width - 0.020), 0.018, 0.003, 0, front_width=0.194)
    for side in (-1, 1):
        z = side * (width / 2 - 0.022)
        arms.tube([(0.0, 0.0, z), (-0.020, 0.025, z), (-0.043, 0.047, z)], 0.010, AXIS_Z)
        brackets.tube([(-0.129, 0.120, z), (-0.152, 0.104, z), (*axis[:2], z)], 0.012, AXIS_Z)
        brackets.beam((*axis[:2], z - 0.013), (*axis[:2], z + 0.013), 0.014)
        brackets.beam((*axis[:2], z + side * 0.012), (*axis[:2], z + side * 0.019), 0.023, sides=20)

    # Positive rotation lowers the forward-facing surface. The last stop is
    # a full half-turn, with the pad facing the ground beneath its own seat.
    angles = (-6, 18, 44, 88, 180)
    rest = mesh_node(doc, binary, name + "_Legrest", {"legrest_shell": tray, "upholstery": pad, "legrest_arm": arms},
                     translation=axis, rotation=rotation(AXIS_Z, angles[0]))
    knee, cover_track = textile_motion(doc, binary, name + "_KneeCover", lambda angle: knee_cover(angle, width / 2 - 0.024), angles)
    clip(doc, binary, name + "_legrest", [
        (rest, "rotation", ADJUSTMENT_STEPS, [rotation(AXIS_Z, a) for a in angles]), cover_track,
    ])
    mount = mesh_node(doc, binary, name + "_LegrestMount", {"legrest_arm": brackets})
    return rest, mount, knee


MATERIALS = {
    "upholstery": ([0.105, 0.145, 0.190, 1.0], 0.0, 0.82),
    "shell": ([0.075, 0.085, 0.095, 1.0], 0.12, 0.62),
    "guardrail": ([0.43, 0.49, 0.55, 1.0], 0.56, 0.42),
    "guard_dock": ([0.098, 0.171, 0.205, 1.0], 0.15, 0.55),
    "guard_release": ([0.85, 0.45, 0.13, 1.0], 0.08, 0.48),
    "harness": ([0.72, 0.40, 0.13, 1.0], 0.0, 0.7),
    "mount_cheek": ([0.155, 0.258, 0.310, 1.0], 0.08, 0.62),
    "mounts": ([0.190, 0.415, 0.475, 1.0], 0.25, 0.40),
    "mount_shoe": ([0.105, 0.210, 0.233, 1.0], 0.44, 0.44),
    "mount_pin": ([0.740, 0.804, 0.833, 1.0], 0.52, 0.34),
    "width_adapter": ([0.63, 0.32, 0.105, 1.0], 0.26, 0.46),
    "legrest_shell": ([0.076, 0.092, 0.107, 1.0], 0.08, 0.63),
    "legrest_arm": ([0.167, 0.223, 0.260, 1.0], 0.24, 0.58),
    "canopy": ([0.115, 0.184, 0.235, 1.0], 0.0, 0.87),
    "canopy_trim": ([0.050, 0.077, 0.099, 1.0], 0.0, 0.76),
    "canopy_stay": ([0.112, 0.177, 0.212, 1.0], 0.10, 0.60),
}
AXIS_X = (1.0, 0.0, 0.0)
AXIS_Z = (0.0, 0.0, 1.0)
LEG_AXIS = (-0.173, 0.087, 0.0)
BACK_PIVOT = (0.091, 0.163, 0.0)
BACK_DEGREES = (95, 115, 135, 155, 175)
BACK_ROTATIONS = tuple(106 - degrees for degrees in BACK_DEGREES)
ADJUSTMENT_STEPS = (0, 0.25, 0.5, 0.75, 1)
CANOPY_EYE = (0.222, 0.552)


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
        buckle = (-0.063, 0.198, side * 0.012)
        lift = 0.036 * smoothstep((-angle - 34) / 35)
        points = [lower]
        for t, rise in ((0.28, lift), (0.56, lift * 0.8)):
            x, y, z = (a + (b - a) * t for a, b in zip(lower, buckle))
            points.append((x, y + rise, z))
        harness.tube([*points, buckle], 0.009, AXIS_Z)
    return {"harness": harness}


def hip_cover(angle):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    x, y = 0.043 - BACK_PIVOT[0], 0.241 - BACK_PIVOT[1]
    end = (BACK_PIVOT[0] + x * c - y * s, BACK_PIVOT[1] + x * s + y * c)
    tilt = math.radians(16 - angle)
    direction = (math.sin(tilt), math.cos(tilt))
    normal = (-direction[1], direction[0])
    start = (0.049, 0.184)
    lead = (start[0] + (end[0] - start[0]) * 0.35, start[1] + 0.003)
    last = tuple(v - t * 0.029 for v, t in zip(end, direction))
    sections = []
    for i in range(13):
        t = i / 12
        u = 1 - t
        px, py = (a * u**3 + 3 * b * t * u**2 + 3 * d * t**2 * u + e * t**3
                  for a, b, d, e in zip(start, lead, last, end))
        nx, ny, _ = unit((normal[0] * t, u + normal[1] * t, 0))
        sections.append((px, py, 0.170, nx, ny))
    return textile_strip(sections)


def add_backrest_canopy(doc, binary, name):
    # Both hinge housings are fixed to the upper shell. The canopy's fabric
    # can deploy independently while inheriting the backrest's full rotation.
    shapes = [canopy_geometry(stop) for stop in range(len(CANOPY_BOW_STOPS))]
    canopy = mesh_node(doc, binary, name + "_Canopy", shapes[-1], origin=BACK_PIVOT, morphs=shapes[:-1])
    weights = [(float(stop == target),) for stop in range(len(shapes)) for target in range(len(shapes) - 1)]
    clip(doc, binary, name + "_canopy", [(canopy, "weights", ADJUSTMENT_STEPS, weights)])

    hinges = Geometry()
    x, y = CANOPY_EYE
    for side in (-1, 1):
        hinges.beam((x, y, side * 0.180), (x, y, side * 0.202), 0.021)
    hinge_node = mesh_node(doc, binary, name + "_CanopyHinges", {"canopy_stay": hinges}, origin=BACK_PIVOT)
    return [canopy, hinge_node]


def handle_extension(doc, binary, level):
    """Telescoping handle motion keyed ahead of the deep rear recline."""
    angle, _ = level
    c, s = math.cos(angle), math.sin(angle)
    # Leave room for the rear canopy as well as the reclined backrest.
    reach = (c * 0.280 - s * 0.216, s * 0.280 + c * 0.216, 0.0)
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


def make_seat(doc, binary, name, anchor, level, rear=False):
    base = {key: Geometry() for key in ("upholstery", "shell", "guard_dock", "harness", "guard_release")}
    soft, shell, harness = (base[key] for key in ("upholstery", "shell", "harness"))
    soft.seat_pad((-0.023, 0.159, 0.0), (0.256, 0.038, 0.374), 0.041, 0.010, 0.013)
    shell.seat_pad((-0.012, 0.126, 0.0), (0.294, 0.034, 0.394), 0.047, 0.007, 0.026)
    for side in (-1, 1):
        soft.box((0.009, 0.199, side * 0.177), (0.180, 0.037, 0.027), math.radians(7))
        harness.beam((0.026, 0.195, side * 0.180), (-0.063, 0.198, side * 0.012), 0.009, sides=8)
        shell.beam((*BACK_PIVOT[:2], side * 0.199), (*BACK_PIVOT[:2], side * 0.203), 0.026, sides=20)
    harness.box((-0.064, 0.196, 0.0), (0.032, 0.032, 0.044))
    harness.beam((-0.115, 0.180, 0.0), (-0.071, 0.200, 0.0), 0.010, sides=8)

    bar = add_guardrail(doc, binary, name, base)

    legrest, legrest_mount, knee = add_legrest(doc, binary, name, rear)

    moving_back = {key: Geometry() for key in ("shell", "upholstery")}
    moving_back["shell"].box((0.151, 0.384, 0.0), (0.062, 0.445, 0.394), -math.radians(16))
    moving_back["upholstery"].box((0.111, 0.391, 0.0), (0.045, 0.425, 0.378), -math.radians(16))
    canopy_nodes = add_backrest_canopy(doc, binary, name)
    backrest = mesh_node(doc, binary, name + "_Backrest", moving_back, origin=BACK_PIVOT, translation=BACK_PIVOT,
                         rotation=rotation(AXIS_Z, BACK_ROTATIONS[0]), children=canopy_nodes)
    straps = mesh_node(doc, binary, name + "_ShoulderStraps", shoulder_straps(BACK_ROTATIONS[0]),
                       morphs=[shoulder_straps(a) for a in BACK_ROTATIONS[1:]])
    hip, hip_track = textile_motion(doc, binary, name + "_HipCover", hip_cover, BACK_ROTATIONS)
    strap_weights = [(1.0 if key == target else 0.0,) for key in range(5) for target in range(1, 5)]
    back_tracks = [
        (backrest, "rotation", ADJUSTMENT_STEPS, [rotation(AXIS_Z, a) for a in BACK_ROTATIONS]),
        (straps, "weights", ADJUSTMENT_STEPS, strap_weights),
        hip_track,
    ]
    if rear:
        back_tracks.extend(handle_extension(doc, binary, level))
    clip(doc, binary, name + "_backrest", back_tracks)

    seat = mesh_node(doc, binary, name + "_Shell", base, translation=(0.0, -SEAT_SHELL_DROP, 0.0),
                     children=[bar, legrest, legrest_mount, knee, backrest, straps, hip])
    mounts = {key: Geometry() for key in ("mount_cheek", "mounts", "mount_shoe", "mount_pin")}
    add_side_mount(mounts["mount_cheek"], mounts["mounts"], mounts["mount_shoe"], mounts["mount_pin"])
    angle, ground = level
    c, s = math.cos(angle), math.sin(angle)
    x, y = anchor
    extras = {"reversibleCabin": True}
    if not rear:
        extras.update(rearLegrestAtFrontFlat=True, frontCanopyClearanceGear=3)
    root = mesh_node(doc, binary, name, mounts, children=[seat],
                     translation=(c*x - s*y, s*x + c*y - ground, 0.0),
                     rotation=rotation(AXIS_Z, math.degrees(angle)), extras=extras)
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(root)


def port_position(centers, port):
    return tuple((a + b) / 2 for a, b in zip(centers[port, 1][:2], centers[port, -1][:2]))


def main():
    output_dir = ROOT / "models/double_cabin"
    output_dir.mkdir(parents=True, exist_ok=True)
    for width in (550, 650):
        for wheel in ("S", "M", "L"):
            doc, binary = prepare_chassis(width, wheel)
            relocate_front_ports(doc, binary, forward=FRONT_PORT_ADVANCE, down=FRONT_PORT_DROP)
            relocate_high_ports(doc, binary, aft=HIGH_PORT_SETBACK, up=HIGH_PORT_RISE)
            centers = standardize_receivers(doc, binary, rise=D02_RISE)
            if width == 650:
                for station in ("L", "H"):
                    add_width_adapter(doc, binary, station, centers)
            level = level_assembly(doc, binary)
            make_seat(doc, binary, "D02_Front", port_position(centers, "L"), level)
            make_seat(doc, binary, "D02_Rear", port_position(centers, "H"), level, rear=True)
            output = output_dir / f"D02_w{width}_state1_wheel_{wheel}.glb"
            write_glb(output, doc, binary)
            print(f"Created {output.relative_to(ROOT)} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
