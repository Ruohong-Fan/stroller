"""Common mounting and cloth shell geometry for the static cabin catalogue."""

from __future__ import annotations

import math

from create_d02_double_seat import (
    AXIS_Z, D02_LAND_Y, Geometry, MATERIALS, STANDARD_MOUNT_Z, add_side_mount,
    add_width_adapter, insert_shoe, mesh_node, port_position, rotation,
)


MATERIALS.update({
    "pet_fabric": ([0.135, 0.230, 0.206, 1], 0, 0.82),
    "pet_mesh": ([0.062, 0.106, 0.100, 1], 0, 0.94),
    "cargo_fabric": ([0.266, 0.235, 0.177, 1], 0, 0.88),
    "cargo_edge": ([0.127, 0.150, 0.144, 1], 0.06, 0.66),
    "bassinet_fabric": ([0.255, 0.315, 0.344, 1], 0, 0.85),
    "liner": ([0.68, 0.68, 0.59, 1], 0, 0.92),
    "carrier_shell": ([0.123, 0.154, 0.183, 1], 0.13, 0.55),
})

WIDE_ADAPTER_CABINS = {"S05", "S06", "S07"}
BODY_MOUNTS = {
    "S01": (0.130, 0.142), "S02": (0.080, 0.118),
    "S03": (0.075, 0.180), "S04": (0.075, 0.226),
    "S05": (0.031, 0.191), "S06": (0.076, 0.171),
    "S07": (0.080, 0.153), "S08": (0.140, 0.213),
    "S09": (0.140, 0.248), "S10": (0.072, 0.256),
    "S11": (0.063, 0.183), "S12": (0.060, 0.248),
    "S13": (0.080, 0.191), "S14": (0.080, 0.245),
    "S15": (0.053, 0.124),
}


def rounded_loop(x, y, z, length, width, radius, *, segments=5):
    """Sample a rounded rectangle in the horizontal plane."""
    dx = length / 2 - radius
    dz = width / 2 - radius
    return [
        (x + sx * dx + radius * math.cos(angle),
         y,
         z + sz * dz + radius * math.sin(angle))
        for corner, (sx, sz) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1)))
        for angle in (
            math.pi / 2 * (corner + i / segments) for i in range(segments + 1)
        )
    ]


def joined_loops(geometry, lower, upper, *, inward=False):
    for i in range(len(lower)):
        j = (i + 1) % len(lower)
        face = (lower[i], upper[i], upper[j], lower[j])
        geometry.quad(*(reversed(face) if inward else face))


def cap_loop(geometry, ring, *, top=False):
    center = tuple(sum(p[axis] for p in ring) / len(ring) for axis in range(3))
    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        geometry.triangle(center, ring[j if top else i], ring[i if top else j])


def shell_loft(geometry, sections, *, thickness=0.010, inner=None):
    """Join shaped cross sections into a lined container with an open top."""
    if inner is None:
        inner = geometry
    outer_rings = [rounded_loop(*section) for section in sections]
    inner_rings = [
        rounded_loop(x, y + thickness if i == 0 else y, z,
                     length - 2 * thickness, width - 2 * thickness,
                     max(0.005, radius - thickness))
        for i, (x, y, z, length, width, radius) in enumerate(sections)
    ]
    for lower, upper in zip(outer_rings, outer_rings[1:]):
        joined_loops(geometry, lower, upper)
    for lower, upper in zip(inner_rings, inner_rings[1:]):
        joined_loops(inner, lower, upper, inward=True)
    joined_loops(geometry, outer_rings[-1], inner_rings[-1])
    cap_loop(geometry, outer_rings[0])
    cap_loop(inner, inner_rings[0], top=True)
    return outer_rings[-1]


def rim_tube(geometry, loop, radius):
    # Short beams close the loop with no discontinuity at the glTF seam.
    for a, b in zip(loop, [*loop[1:], loop[0]]):
        geometry.beam(a, b, radius, sides=8)


def group(*materials):
    return {material: Geometry() for material in materials}


def attach(doc, binary, name, groups, anchor, level, **metadata):
    """Place a static module authored in the chassis coordinate system."""
    angle, ground = level
    c, s = math.cos(angle), math.sin(angle)
    x, y = anchor
    node = mesh_node(doc, binary, name, groups,
                     translation=(c*x - s*y, s*x + c*y - ground, 0.0),
                     rotation=rotation(AXIS_Z, math.degrees(angle)),
                     **({"extras": metadata} if metadata else {}))
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(node)
    return node


def mount_pair(groups, span, target, *, shoe=True):
    """Use two centered shoes with low, thick returns to the body rails."""
    dx, dy, half_body = target
    for side in (-1, 1):
        if shoe:
            insert_shoe(groups["mount_shoe"], groups["mount_pin"], side, span)
        pickup = (0, D02_LAND_Y + 0.011, side * span)
        landing = (dx, dy, side * half_body)
        groups["mounts"].box(pickup, (0.060, 0.022, 0.034))
        if math.dist(pickup, landing) > 0.010:
            groups["mount_cheek"].beam(pickup, landing, 0.023, sides=12)
        groups["mount_cheek"].box(landing, (0.075, 0.027, 0.033))


def add_static_cabin(doc, binary, name, cabin_id, width, centers, level, station,
                     *, shift=(0, 0), profile="standard"):
    from catalog_cabin_geometry import cabin_geometry

    mount = port_position(centers, station)
    span = abs(centers[station, 1][2])
    if width == 650 and cabin_id in WIDE_ADAPTER_CABINS:
        add_width_adapter(doc, binary, station, centers)
        span = STANDARD_MOUNT_Z

    dx, dy = shift
    body_groups = cabin_geometry(cabin_id, width, profile)
    mounts = group("mount_cheek", "mounts", "mount_shoe", "mount_pin")
    if cabin_id == "S05" and not any(shift):
        add_side_mount(*(mounts[key] for key in ("mount_cheek", "mounts", "mount_shoe", "mount_pin")))
    else:
        pickup_y, pickup_z = BODY_MOUNTS[cabin_id]
        if cabin_id == "S15" and profile == "paired":
            pickup_z = 0.105
        mount_pair(mounts, span, (dx, dy + pickup_y, pickup_z))
    attach(doc, binary, name + "_Mount_" + station, mounts, mount, level, station=station)
    return attach(doc, binary, name, body_groups, (mount[0] + dx, mount[1] + dy),
                  level, cabinId=cabin_id, station=station, profile=profile)


def add_aux_mount(doc, binary, name, centers, level, station, target):
    mount = port_position(centers, station)
    mounts = group("mount_cheek", "mounts", "mount_shoe", "mount_pin")
    x, y, half_width = target
    mount_pair(mounts, abs(centers[station, 1][2]),
               (x - mount[0], y - mount[1], half_width))
    return attach(doc, binary, name, mounts, mount, level, station=station)
