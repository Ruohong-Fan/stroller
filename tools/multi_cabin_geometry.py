"""Twin-seat and camping cabin modules in station coordinates."""

from __future__ import annotations

import math
from collections import defaultdict

from create_d02_double_seat import Geometry


def _add_child_seat(groups, center, width, *, faces_rear=False, footrest=False):
    """Build a padded seat and turn the complete insert toward the footwell."""
    parts = defaultdict(Geometry)
    shell, cloth = parts["shell"], parts["upholstery"]
    shell.seat_pad((-0.020, 0.040, 0), (0.268, 0.041, width), 0.037, 0.008, 0.014)
    cloth.seat_pad((-0.024, 0.070, 0), (0.246, 0.035, width - 0.018), 0.034, 0.008, 0.017)

    hip_x, hip_y = 0.095, 0.072
    lean = math.radians(12)
    rise_x, rise_y = math.sin(lean), math.cos(lean)

    def back_point(height, front, z):
        return hip_x + height * rise_x - front * rise_y, hip_y + height * rise_y + front * rise_x, z

    for material, thickness, pad_width, height, offset in (
        ("shell", 0.036, width, 0.315, -0.009),
        ("upholstery", 0.032, width - 0.018, 0.296, 0.015),
    ):
        cushion = Geometry()
        cushion.seat_pad((height / 2, offset, 0), (height, thickness, pad_width), 0.037, 0.007, 0)
        parts[material].vertices.extend(back_point(*v) for v in cushion.vertices)
        parts[material].normals.extend(
            (x * rise_x - y * rise_y, x * rise_y + y * rise_x, z)
            for x, y, z in cushion.normals
        )

    harness = parts["harness"]
    buckle = (-0.078, 0.096, 0)
    for side in (-1, 1):
        strap_route = [
            back_point(0.232, 0.034, side * width * 0.23),
            back_point(0.066, 0.037, side * width * 0.14),
            (-0.011, 0.102, side * 0.019),
            buckle,
        ]
        harness.tube(strap_route, 0.008, (0, 0, 1), sides=8)
        harness.beam((0.039, 0.102, side * (width / 2 - 0.014)), buckle, 0.008, sides=8)
    harness.box(buckle, (0.028, 0.021, 0.036))

    # Each child has a separate bar, with room between the two middle pins.
    half = width / 2 - 0.016
    guard = [(-0.014, 0.167, -half), (-0.088, 0.181, -half),
             (-0.124, 0.174, -half + 0.025), (-0.124, 0.174, half - 0.025),
             (-0.088, 0.181, half), (-0.014, 0.167, half)]
    parts["guardrail"].tube(guard, 0.011, (0, 1, 0), sides=10)
    for side in (-1, 1):
        parts["guard_dock"].beam((-0.014, 0.063, side * half), (-0.014, 0.168, side * half), 0.015)
        parts["guard_release"].beam((-0.014, 0.156, side * (half + 0.009)),
                                    (-0.014, 0.156, side * (half + 0.017)), 0.006)

    if footrest:
        shell.seat_pad((-0.184, 0.024, 0), (0.167, 0.034, width * 0.84), 0.030, 0.007, 0.013)
        cloth.seat_pad((-0.191, 0.048, 0), (0.147, 0.025, width * 0.76), 0.027, 0.006, 0.017)
        for side in (-1, 1):
            lateral = side * width * 0.33
            parts["legrest_arm"].beam((-0.124, 0.027, lateral), (-0.225, 0.022, lateral), 0.012)

    cx, cy, cz = center
    facing = -1 if faces_rear else 1
    for material, geometry in parts.items():
        groups[material].vertices.extend((cx + facing * x, cy + y, cz + facing * z)
                                         for x, y, z in geometry.vertices)
        groups[material].normals.extend((facing * x, y, facing * z) for x, y, z in geometry.normals)


def _twin_seats(groups):
    # The two seats share a load pan and moulded side rails.
    groups["shell"].seat_pad((-0.036, 0.018, 0), (0.424, 0.046, 0.602), 0.061, 0.010, 0.012)
    for side in (-1, 1):
        z = side * 0.147
        _add_child_seat(groups, (0, 0.020, z), 0.279, footrest=True)
        groups["mount_cheek"].side_cheek(
            [(-0.206, 0.025), (-0.127, 0.097), (0.111, 0.123),
             (0.168, 0.069), (0.141, 0.006), (-0.165, -0.004)],
            side * 0.291, 0.029,
        )
    groups["canopy_sleeve"].seat_pad((0.014, 0.074, 0), (0.210, 0.035, 0.033), 0.014, 0.005, 0.011)
    groups["legrest_arm"].beam((0.145, 0.053, -0.283), (0.145, 0.053, 0.283), 0.019)


def _wagon_rim(half_length, half_width, corner, height, *, front_row, floor=False):
    points = []
    corners = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    center = -0.159 if front_row else -0.073
    half_length -= 0.123 if front_row else 0.024
    for quadrant, (sx, sz) in enumerate(corners):
        for step in range(7):
            angle = (quadrant + step / 6) * math.pi / 2
            x = center + sx * (half_length - corner) + corner * math.cos(angle)
            z = sz * (half_width - corner) + corner * math.sin(angle)
            points.append((x, z))
    ramp_start = -0.010 if front_row else 0.025

    # Carry the rear floor over the rising frame tubes. The tandem rim stays
    # lower to preserve the loading opening below the upper cabin.
    def lifted(x, z):
        if floor:
            y = height + (0.079 / 0.175 if front_row else 0.640) * max(0, x - ramp_start)
        elif front_row:
            y = height - 0.039 * max(0, min(1, (x + 0.155) / 0.225))
        else:
            y = height + 0.390 * max(0, x - ramp_start)
        return x, y, z

    ring = []
    for (x, z), (nx, nz) in zip(points, [*points[1:], points[0]]):
        ring.append(lifted(x, z))
        if abs(nx - x) > 0.15:
            t = (ramp_start - x) / (nx - x)
            ring.append(lifted(ramp_start, z + t * (nz - z)))
    return ring


def _wagon(groups, width, twin_rows, front_row):
    half_width = 0.293 if twin_rows else 0.247 if width == 550 else 0.276
    footwell = groups["shell"]
    floor_bottom = _wagon_rim(0.420, half_width - 0.021, 0.057, 0.013, front_row=front_row, floor=True)
    floor_top = _wagon_rim(0.423, half_width - 0.016, 0.059, 0.035, front_row=front_row, floor=True)
    for i in range(len(floor_bottom)):
        j = (i + 1) % len(floor_bottom)
        footwell.quad(floor_bottom[i], floor_top[i], floor_top[j], floor_bottom[j])
    for ring, top in ((floor_bottom, False), (floor_top, True)):
        side = [point for point in ring if point[2] > 0]
        for a, b in zip(side, side[1:]):
            strip = (a, b, (b[0], b[1], -b[2]), (a[0], a[1], -a[2]))
            footwell.quad(*(reversed(strip) if top else strip))

    # A shallow sewn return covers the top edge on the inside and outside.
    wall = groups["canopy_sleeve"]
    lower = _wagon_rim(0.418, half_width - 0.020, 0.057, 0.026, front_row=front_row, floor=True)
    upper = _wagon_rim(0.455, half_width, 0.063, 0.206, front_row=front_row)
    inner = _wagon_rim(0.448, half_width - 0.007, 0.057, 0.201, front_row=front_row)
    liner = _wagon_rim(0.410, half_width - 0.027, 0.051, 0.031, front_row=front_row, floor=True)
    for inside, outside in ((lower, upper), (upper, inner), (inner, liner)):
        for i in range(len(inside)):
            j = (i + 1) % len(inside)
            wall.quad(inside[i], outside[i], outside[j], inside[j])
    groups["guardrail"].tube([*upper, upper[0]], 0.009, (0, 1, 0), sides=10, capped=False)

    for side in (-1, 1):
        z = side * (half_width - 0.033)
        path = [(-0.381, 0.028, z), (-0.010 if front_row else 0.025, 0.028, z)]
        if not front_row:
            path.append((0.249, 0.171, z))
        groups["legrest_arm"].tube(path, 0.013, (0, 0, 1), sides=12)
        rear_foot = (-0.036, 0.017) if front_row else (0.188, 0.112)
        for x, y in ((-0.308, 0.002), rear_foot):
            groups["mounts"].seat_pad((x, y, z), (0.080, 0.039, 0.060), 0.018, 0.006, 0)

    rows = [(-0.283, 0.028, True)]
    if not front_row:
        rows.append((0.141, 0.147, False))
    supports = groups["mount_cheek"]
    for x, y, rearward in rows:
        for z in (-0.137, 0.137) if twin_rows else (0,):
            seat_width = 0.253 if twin_rows else 0.326
            if rearward:
                supports.box((x, 0.040, z), (0.143, 0.027, seat_width - 0.041))
            else:
                supports.box((x - 0.063, 0.121, z), (0.058, 0.107, seat_width - 0.035))
                supports.box((x + 0.063, 0.155, z), (0.061, 0.050, seat_width - 0.032))
            _add_child_seat(groups, (x, y, z), seat_width, faces_rear=rearward)


def multi_cabin_geometry(cabin_id, width, profile="standard"):
    """Return material-group geometry for S10, S11 or S12."""
    groups = defaultdict(Geometry)
    if cabin_id == "S10":
        _twin_seats(groups)
    elif cabin_id in ("S11", "S12"):
        _wagon(groups, width, twin_rows=cabin_id == "S12", front_row=profile == "front-row")
    else:
        raise ValueError(f"No multi-seat geometry for {cabin_id}")
    return dict(groups)
