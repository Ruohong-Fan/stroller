"""Place the D02 lower receiver pair on the front rails of the display chassis."""

from __future__ import annotations

import math
import struct

from generate_wheel_variants import position_accessor_for_node, read_positions
from prepare_d02_chassis import bounds, replace_positions


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _outline(points):
    """Return the counterclockwise outline of the two gusset bearing pads."""
    ordered = sorted(set(points))

    def half(path):
        edge = []
        for point in path:
            while len(edge) >= 2 and _cross(edge[-2], edge[-1], point) <= 0:
                edge.pop()
            edge.append(point)
        return edge

    return half(ordered)[:-1] + half(reversed(ordered))[:-1]


def _plate(doc, binary, name, outline, middle_z, thickness, material):
    lower = [(x, y, middle_z - thickness / 2) for x, y in outline]
    upper = [(x, y, middle_z + thickness / 2) for x, y in outline]
    vertices, normals = [], []

    def triangle(a, b, c):
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        length = math.hypot(*normal)
        if length:
            vertices.extend((a, b, c))
            normals.extend([tuple(n / length for n in normal)] * 3)

    for i in range(1, len(outline) - 1):
        triangle(lower[0], lower[i + 1], lower[i])
        triangle(upper[0], upper[i], upper[i + 1])
    for i in range(len(outline)):
        j = (i + 1) % len(outline)
        triangle(lower[i], lower[j], upper[j])
        triangle(lower[i], upper[j], upper[i])

    attributes = {}
    for semantic, values in (("POSITION", vertices), ("NORMAL", normals)):
        binary.extend(b"\x00" * (-len(binary) % 4))
        offset = len(binary)
        for value in values:
            binary.extend(struct.pack("<3f", *value))
        view = len(doc["bufferViews"])
        doc["bufferViews"].append({
            "buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset, "target": 34962,
        })
        accessor = {"bufferView": view, "componentType": 5126, "count": len(values), "type": "VEC3"}
        if semantic == "POSITION":
            accessor["min"] = [min(axis) for axis in zip(*values)]
            accessor["max"] = [max(axis) for axis in zip(*values)]
        attributes[semantic] = len(doc["accessors"])
        doc["accessors"].append(accessor)

    mesh = len(doc["meshes"])
    doc["meshes"].append({"name": name, "primitives": [{"attributes": attributes, "material": material}]})
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(len(doc["nodes"]))
    doc["nodes"].append({"name": name, "mesh": mesh})
    doc["buffers"][0]["byteLength"] = len(binary)


def _bearing_pads(rail_points, receiver):
    # Infer the installed rail direction from the actual CAD vertices. Work
    # in its side plane so both width variants use their native geometry.
    cx = sum(p[0] for p in rail_points) / len(rail_points)
    cy = sum(p[1] for p in rail_points) / len(rail_points)
    xx = sum((p[0] - cx) ** 2 for p in rail_points)
    yy = sum((p[1] - cy) ** 2 for p in rail_points)
    xy = sum((p[0] - cx) * (p[1] - cy) for p in rail_points)
    angle = math.atan2(2 * xy, xx - yy) / 2
    c, s = math.cos(angle), math.sin(angle)

    def project(x, y):
        return (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c

    rail_u = [project(x, y)[0] for x, y, _ in rail_points]
    first, last = min(rail_u), max(rail_u)
    u, v = project(*receiver[:2])
    reach = 0.046
    if abs(v) <= 0.006 and first + reach <= u <= last - reach:
        # A receiver shifted along the native tube is still supported by it.
        return None

    anchor = min(max(u, first + reach), last - reach)
    # The receiver pad is under the socket. The other end spreads out along
    # the native lower tube instead of balancing the receiver on a thin rod.
    rail_pad = [(anchor + a, b) for a in (-reach, reach) for b in (-0.010, 0.009)]
    receiver_pad = [(u + a, v + b) for a in (-0.043, 0.043) for b in (-0.017, -0.002)]
    profile = [(cx + a * c - b * s, cy + a * s + b * c) for a, b in rail_pad + receiver_pad]
    return _outline(profile)


def _relocate_ports(doc, binary, station, rail_name, support_name, dx, dy):
    if not (dx or dy):
        return
    for side in (1, -1):
        receiver = next(n for n in doc["nodes"] if n["name"].startswith(f"narrow-open_host_{side}_{station}."))
        rail = next(n for n in doc["nodes"] if n["name"].startswith(f"{rail_name}_{side}."))
        pos = position_accessor_for_node(doc, receiver)
        replace_positions(doc, binary, pos, lambda p: (p[0] + dx, p[1] + dy, p[2]))
        lo, hi = bounds(doc, receiver)
        center = tuple((a + b) / 2 for a, b in zip(lo, hi))
        rail_points = read_positions(doc, binary, position_accessor_for_node(doc, rail))
        outline = _bearing_pads(rail_points, center)
        if outline is None:
            continue
        rail_min, rail_max = bounds(doc, rail)
        material = doc["meshes"][rail["mesh"]]["primitives"][0]["material"]
        pad_center, pad_thickness = center[2], rail_max[2] - rail_min[2] - 0.002
        if station == "H":
            # Reinforce the outside wall without occupying the handle's
            # telescoping channel inside the upper tube.
            pad_center += math.copysign(0.020, center[2])
            pad_thickness = 0.014
        _plate(
            doc, binary, f"{support_name}_{side}", outline, pad_center, pad_thickness, material,
        )


def relocate_front_ports(doc: dict, binary: bytearray, *, forward: float = 0.0, down: float = 0.0) -> None:
    """Move both low receiver datums before fitting and leveling the display assembly."""
    _relocate_ports(doc, binary, "L", "main_lower", "D02_FrontLowerReceiverGusset", -forward, -down)


def relocate_high_ports(doc: dict, binary: bytearray, *, aft: float = 0.0, up: float = 0.0) -> None:
    """Move both high receiver datums and support them on the native upper rail."""
    _relocate_ports(doc, binary, "H", "main_upper", "D02_RearUpperReceiverGusset", aft, up)
