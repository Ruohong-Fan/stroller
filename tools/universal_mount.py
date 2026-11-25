"""Display geometry and mating dimensions for the reversible cabin mount."""

from __future__ import annotations

import math
import re
import struct

from generate_wheel_variants import position_accessor_for_node, read_positions


FRAME_LAND_Y = 0.082
RECEIVER_OUTER_X = 0.064
RECEIVER_OUTER_Z = 0.048
POCKET_X = 0.046
POCKET_Z = 0.030
POCKET_BOTTOM_Y = FRAME_LAND_Y - 0.015
INSERT_X = 0.044
INSERT_Z = 0.028
INSERT_DEPTH = 0.012
LOCK_Y = FRAME_LAND_Y - 0.006
LOCK_CLEARANCE = 0.005  # Half-span of the pin opening in X and Y.
WIDE_RAIL_OFFSET = 0.036

_HOST = re.compile(r"^narrow-open_host_(-?1)_([HML])(?:\.|$)")
_NAME = "Universal_Cabin_Receiver_"


def _hosts(doc):
    scene = doc["scenes"][doc.get("scene", 0)]
    found = {}
    for index in scene["nodes"]:
        node = doc["nodes"][index]
        match = _HOST.match(node.get("name", ""))
        if match:
            side, station = match.groups()
            found[station, int(side)] = node
    if len(found) != 6:
        raise ValueError("The chassis must contain both H, M and L receiver datums")
    return found


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _points(doc, binary, node):
    positions = read_positions(doc, binary, position_accessor_for_node(doc, node))
    matrix = node.get("matrix")
    if matrix is not None:
        return [tuple(sum(matrix[row + 4 * k] * p[k] for k in range(3)) + matrix[12 + row]
                      for row in range(3)) for p in positions]
    scale = node.get("scale", (1.0, 1.0, 1.0))
    offset = node.get("translation", (0.0, 0.0, 0.0))
    *axis, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    result = []
    for p in positions:
        value = tuple(a * b for a, b in zip(p, scale))
        cross = tuple(2 * x for x in _cross(axis, value))
        r = _cross(axis, cross)
        result.append(tuple(v + w * t + d + o for v, t, d, o in zip(value, cross, r, offset)))
    return result


def _center(points):
    return tuple((min(axis) + max(axis)) / 2 for axis in zip(*points))


def native_port_centers(doc, binary):
    """The six anchor centers in native chassis coordinates, in meters."""
    return {key: _center(_points(doc, binary, node)) for key, node in _hosts(doc).items()}


def _rail_offset(z):
    return math.copysign(WIDE_RAIL_OFFSET, z) if abs(z) > 0.170 else 0.0


def receiver_port_centers(doc, binary):
    """Mating datum (station, side) centers before chassis leveling, excluding land height."""
    return {key: (x, y, z + _rail_offset(z)) for key, (x, y, z) in native_port_centers(doc, binary).items()}


def _fold_rotation(points, rest):
    if len(points) != len(rest):
        raise ValueError("Cannot match folded receiver datum to its open CAD topology")
    x, y, _ = _center(points)
    rx, ry, _ = _center(rest)
    dot, cross = 0.0, 0.0
    for p, q in zip(points, rest):
        ax, ay = p[0] - x, p[1] - y
        bx, by = q[0] - rx, q[1] - ry
        dot += bx * ax + by * ay
        cross += bx * ay - by * ax
    return math.atan2(cross, dot)


def _box(vertices, normals, x, y, z):
    points = {(a, b, c): (x[a], y[b], z[c]) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    faces = (
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )
    for normal, corners in faces:
        vertices.extend(points[corners[index]] for index in (0, 1, 2, 0, 2, 3))
        normals.extend([normal] * 6)


def receiver_geometry(rail_offset=0.0, *, include_support=True, rise=0.0, native_low=False):
    """Return receiver geometry and the fixed saddle reaching its native rail."""
    v, n = [], []
    xo, zo = RECEIVER_OUTER_X / 2, RECEIVER_OUTER_Z / 2
    xi, zi = POCKET_X / 2, POCKET_Z / 2
    floor = POCKET_BOTTOM_Y + rise
    land = FRAME_LAND_Y + rise
    pin = LOCK_Y + rise
    rim = (floor, land)
    # The low saddle spans to the stock bearing pad. The raised collar is
    # outboard of the rail on the 550 and 650 frames so its pocket stays open.
    if include_support:
        # Bear against the outside wall, leaving the telescoping tube's
        # 11.5 mm radius free inside the upper rail at the high station.
        if rail_offset:
            inner = -rail_offset + math.copysign(0.0125, rail_offset)
            outer = math.copysign(0.014, rail_offset)
            # At the stock low station the rear leg folds beside the rail.
            # The bracket takes the space ahead of that leg's closing path.
            x, y = ((-0.035, -0.012), (-0.003, 0.027)) if native_low else ((-0.022, 0.022), (-0.011, 0.022))
            _box(v, n, x, y, (min(inner, outer), max(inner, outer)))
            x, base = ((-0.030, 0.012), 0.025) if native_low else ((-0.022, 0.022), 0.020)
            _box(v, n, x, (base, floor - 0.006), (-0.018, 0.018))
        else:
            wing = 0.017 if native_low else 0.023
            for wall in ((-wing, -0.0125), (0.0125, wing)):
                _box(v, n, (-0.022, 0.022), (-0.011, floor - 0.006), wall)
            if native_low:
                _box(v, n, (-0.022, 0.022), (0.043, floor - 0.006), (-0.023, 0.023))
    _box(v, n, (-xo, xo), (floor - 0.008, floor), (-zo, zo))
    for x in ((-xo, -xi), (xi, xo)):
        _box(v, n, x, rim, (-zo, zo))
    for z in ((-zo, -zi), (zi, zo)):
        for x in ((-xi, -LOCK_CLEARANCE), (LOCK_CLEARANCE, xi)):
            _box(v, n, x, rim, z)
        for y in ((floor, pin - LOCK_CLEARANCE), (pin + LOCK_CLEARANCE, land)):
            _box(v, n, (-LOCK_CLEARANCE, LOCK_CLEARANCE), y, z)
    return v, n


def _accessor(doc, binary, values, semantic):
    binary.extend(b"\x00" * (-len(binary) % 4))
    byte_offset = len(binary)
    for value in values:
        binary.extend(struct.pack("<3f", *value))
    view = len(doc["bufferViews"])
    doc["bufferViews"].append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(binary)-byte_offset,
                               "target": 34962})
    data = {"bufferView": view, "componentType": 5126, "count": len(values), "type": "VEC3"}
    if semantic == "POSITION":
        data["min"] = [min(axis) for axis in zip(*values)]
        data["max"] = [max(axis) for axis in zip(*values)]
    out = len(doc["accessors"])
    doc["accessors"].append(data)
    return out


def remove_receiver_nodes(doc):
    """Remove active receiver bodies before rebuilding them on relocated datums."""
    scene = doc["scenes"][doc.get("scene", 0)]
    scene["nodes"] = [index for index in scene["nodes"] if not doc["nodes"][index].get("name", "").startswith(_NAME)]


def standardize_receivers(doc, binary, *, reference=None, rise=0.0):
    """Fit six mounts on the base frame. For a folded CAD pose pass its open (doc, binary).

    Call after any datum moves and before D02's final chassis leveling. The
    native host blocks remain as the load pads and provide stable anchors.
    Re-running on a standardized chassis replaces the previous receiver nodes.
    """
    hosts = _hosts(doc)
    rest_doc, rest_binary = reference or (doc, binary)
    rest = _hosts(rest_doc)
    remove_receiver_nodes(doc)
    scene = doc["scenes"][doc.get("scene", 0)]
    chassis_names = {doc["nodes"][index].get("name", "") for index in scene["nodes"]}
    material_name = "Universal Cabin Receiver"
    material = next((i for i, m in enumerate(doc["materials"]) if m.get("name") == material_name), None)
    if material is None:
        material = len(doc["materials"])
        doc["materials"].append({"name": material_name, "pbrMetallicRoughness": {
            "baseColorFactor": [0.18, 0.26, 0.30, 1], "metallicFactor": 0.58, "roughnessFactor": 0.42,
        }})
    centers = {}
    for (station, side), node in hosts.items():
        points = _points(doc, binary, node)
        x, y, rail_z = _center(points)
        offset = _rail_offset(rail_z)
        z = rail_z + offset
        centers[station, side] = (x, y, z)
        native_low = station == "L" and f"D02_FrontLowerReceiverGusset_{side}" not in chassis_names
        local_vertices, local_normals = receiver_geometry(offset, rise=rise, native_low=native_low)
        angle = _fold_rotation(points, _points(rest_doc, rest_binary, rest[station, side]))
        c, s = math.cos(angle), math.sin(angle)
        vertices = [(x + a*c - b*s, y + a*s + b*c, z + d) for a, b, d in local_vertices]
        normals = [(a*c - b*s, a*s + b*c, d) for a, b, d in local_normals]
        attributes = {k: _accessor(doc, binary, p, k) for k, p in (("POSITION", vertices), ("NORMAL", normals))}
        name = f"{_NAME}{station}_{side}"
        mesh = len(doc["meshes"])
        doc["meshes"].append({"name": name, "primitives": [{"attributes": attributes, "material": material}]})
        scene["nodes"].append(len(doc["nodes"]))
        doc["nodes"].append({"name": name, "mesh": mesh, "extras": {
            "station": station, "side": side, "nativeDatum": [x, y, z], "railDatum": [x, y, rail_z],
            "nativeLand": [x - s * (FRAME_LAND_Y + rise), y + c * (FRAME_LAND_Y + rise), z],
        }})
    doc["buffers"][0]["byteLength"] = len(binary)
    return centers
