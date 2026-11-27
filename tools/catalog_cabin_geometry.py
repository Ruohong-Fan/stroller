"""Cabin bodies for the first assembled previews of the stroller catalogue."""

from __future__ import annotations

from functools import lru_cache
import math
import struct

from catalog_modeling import group, rim_tube, shell_loft
from create_d02_double_seat import ROOT, compose_rotations, cross
from generate_wheel_variants import read_glb, read_positions


def canopy(groups, sections, fabric="canopy"):
    """Sew a continuous hood between shaped transverse bows."""
    rows = [
        [(x, y + arch * math.sin(t * math.pi / 16), -half_width * math.cos(t * math.pi / 16))
         for t in range(17)]
        for x, y, half_width, arch in sections
    ]
    for front, rear in zip(rows, rows[1:]):
        for i in range(len(front) - 1):
            groups[fabric].quad(front[i], front[i + 1], rear[i + 1], rear[i])
    for row in rows:
        groups["canopy_rib"].tube(row, 0.0045, (1, 0, 0), sides=8)
    for start in (0, -1):
        groups["canopy_rib"].tube([row[start] for row in rows], 0.004, (0, 0, 1), sides=8)


def rounded_pad(geometry, x, y, length, width, *, depth=0.033, radius=0.043,
                front_width=None, pitch=0):
    before = len(geometry.vertices)
    geometry.seat_pad((x, y, 0), (length, depth, width), radius, 0.007, 0.004,
                      front_width=front_width)
    if pitch:
        c, s = math.cos(pitch), math.sin(pitch)
        geometry.vertices[before:] = [
            (x + (a-x)*c - (b-y)*s, y + (a-x)*s + (b-y)*c, z)
            for a, b, z in geometry.vertices[before:]
        ]
        geometry.normals[before:] = [(a*c-b*s, a*s+b*c, z) for a, b, z in geometry.normals[before:]]


def pet_cabin(cabin_id, low):
    length, width, center, rim_height, top = {
        "S02": (0.362, 0.304, -0.025, 0.155, 0.315),
        "S03": (0.620, 0.441, -0.055, 0.181, 0.388),
        "S04": (0.706, 0.543, -0.060, 0.199, 0.422),
    }[cabin_id]
    parts = group("pet_fabric", "pet_mesh", "upholstery", "canopy_rib", "guard_release")
    height = rim_height + 0.028 if low else rim_height
    rim = shell_loft(parts["pet_fabric"], [
        (center, 0.045, 0, length - 0.061, width - 0.042, 0.050),
        (center, 0.070, 0, length - 0.026, width - 0.026, 0.065),
        (center, height, 0, length, width, 0.075),
    ], inner=parts["upholstery"])
    rim_tube(parts["pet_mesh"], rim, 0.005)
    rounded_pad(parts["upholstery"], center, 0.070, length - 0.079, width - 0.059, depth=0.023)
    if low:
        return parts

    fore, aft = center - length / 2 + 0.056, center + length / 2 - 0.055
    half = width * 0.491
    canopy(parts, [
        (fore, height + 0.051, half * .89, (top-height)*.48),
        (fore + length * .21, height + 0.077, half * .87, top-height-.077),
        (aft, height + .049, half * .84, (top-height)*.48),
    ], fabric="pet_fabric")
    mesh = parts["pet_mesh"]
    for side in (-1, 1):
        for i in range(12):
            x = fore + (aft - fore) * i / 11
            mesh.beam((x, height, side * half), (x, height + .054, side * half * .867), .0017, sides=5)
        for i in range(1, 5):
            t = i / 5
            mesh.beam((fore, height+.053*t, side*half*(1-.112*t)),
                      (aft, height+.049*t, side*half*(1-.147*t)), .0017, sides=5)
        mesh.beam((fore, height, side * half), (fore, height+.055, side*half*.89), .005)
        mesh.beam((aft, height, side * half), (aft, height+.050, side*half*.84), .005)
        parts["guard_release"].beam((fore + .025, height + .047, side * half * .9),
                                    (fore + .025, height + .035, side * half * .93), .003, sides=6)
    for i in range(12):
        t = -0.87 + 1.74 * i / 11
        mesh.beam((fore - 0.022, height, t * half), (fore, height+.052, t*half*.89), .0017, sides=5)
    return parts


def bassinet(low):
    parts = group("bassinet_fabric", "liner", "canopy", "canopy_rib", "shell", "guardrail")
    x = -0.027
    rim = shell_loft(parts["bassinet_fabric"], [
        (x, .045, 0, .608, .329, .109),
        (x, .081, 0, .682, .369, .126),
        (x, .210, 0, .715, .394, .146),
    ], thickness=.013, inner=parts["liner"])
    rim_tube(parts["canopy_rib"], rim, .005)
    rounded_pad(parts["liner"], x, .086, .608, .327, radius=.096)
    if low:
        roof = [(.178, .211, .177, .043), (.232, .211, .168, .057), (.299, .207, .124, .024)]
    else:
        roof = [(.032, .213, .193, .192), (.142, .211, .190, .222),
                (.244, .212, .171, .177), (.305, .208, .109, .088), (.323, .194, .032, .025)]
    canopy(parts, roof)
    for side in (-1, 1):
        parts["guardrail"].beam((-.128, .164, side * .192), (-.028, .166, side * .194), .008)
    return parts


def infant_carrier(low):
    parts = group("carrier_shell", "upholstery", "liner", "guardrail", "guard_dock",
                  "guard_release", "harness", "canopy", "canopy_rib")
    rim = shell_loft(parts["carrier_shell"], [
        (.004, .033, 0, .410, .268, .085),
        (-.004, .087, 0, .514, .322, .128),
        (-.011, .187, 0, .540, .354, .132),
    ], thickness=.014, inner=parts["upholstery"])
    rim_tube(parts["guard_dock"], rim, .006)
    rounded_pad(parts["liner"], -.056, .108, .352, .265, depth=.028, radius=.048, pitch=.14)
    rounded_pad(parts["upholstery"], .100, .200, .228, .267, depth=.058, radius=.053, pitch=.66)
    rounded_pad(parts["liner"], .165, .253, .119, .214, depth=.052, pitch=.62)
    for side in (-1, 1):
        parts["harness"].tube([(.170, .280, side*.065), (.097, .209, side*.076),
                               (-.048, .138, side*.019)], .008, (0, 1, 0), sides=8)
        parts["guard_dock"].beam((-.012, .180, side * .169), (-.012, .180, side * .183), .026)
        parts["guard_release"].beam((-.012, .180, side * .185), (-.012, .180, side * .191), .013)
    parts["harness"].box((-.049, .145, 0), (.035, .021, .040))
    handle = [(0 if low else -.012, .180, -.184),
              (.121 if low else -.014, .228 if low else .345, -.176),
              (.234 if low else -.021, .232 if low else .435, -.141),
              (.247 if low else -.026, .240 if low else .467, -.064)]
    handle += [(a, b, -c) for a, b, c in reversed(handle)]
    parts["guardrail"].tube(handle, .010, (1, 0, 0), sides=10)
    parts["guard_dock"].beam(handle[3], handle[4], .014, sides=10)
    canopy(parts, [(.086 if low else .033, .195, .153, .105 if low else .144),
                   (.171, .206, .146, .116 if low else .165),
                   (.241, .203, .114, .077), (.254, .186, .061, .034)])
    return parts


def cargo_cabin(cabin_id, paired=False):
    length, width, center, bottom, top = {
        "S13": (.600, .450, -.050, .043, .222),
        "S14": (.670, .546, -.065, .040, .238),
        "S15": (.318, .278, -.020, -.069, .086),
    }[cabin_id]
    soft = cabin_id == "S15"
    if soft and paired:
        width = 0.232
    parts = group("cargo_fabric", "cargo_edge", "shell", "harness", "guard_release")
    lip = shell_loft(parts["cargo_fabric"], [
        (center, bottom, 0, length - (.082 if soft else .055), width - .053, .041),
        (center, bottom+.024, 0, length - .032, width - .028, .046),
        (center, top, 0, length, width, .054),
    ], thickness=.008, inner=parts["cargo_edge"])
    rim_tube(parts["cargo_edge"], lip, .005 if soft else .009)
    for side in (-1, 1):
        for x in (center-length*.27, center+length*.27):
            geometry = parts["harness" if soft else "cargo_edge"]
            geometry.tube([(x, bottom+.012, side*(width/2-.024)),
                           (x, bottom+.048, side*(width/2-.011)),
                           (x, top+.002, side*(width/2-.003))],
                          .006 if soft else .007, (1, 0, 0), sides=8)
        parts["shell"].beam((center-length*.24, bottom+.015, side*width*.32),
                            (center+length*.24, bottom+.015, side*width*.32), .008, sides=8)
        parts["guard_release"].box((center+.038, top+.002, side*(width/2-.011)), (.043, .010, .024))
    return parts


def rotate_vec(quaternion, vector):
    q = quaternion[:3]
    a = cross(q, vector)
    b = cross(q, a)
    return tuple(vector[i] + 2 * (quaternion[3]*a[i] + b[i]) for i in range(3))


def seat_keyframe(doc, binary, animation, time):
    """Read an exact authored detent from D02's packed float animation tracks."""
    clip = next(clip for clip in doc["animations"] if clip["name"] == animation)

    def values(index):
        accessor = doc["accessors"][index]
        view = doc["bufferViews"][accessor["bufferView"]]
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        dimensions = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
        return struct.unpack_from(f"<{accessor['count'] * dimensions}f", binary, offset)

    pose = {}
    for channel in clip["channels"]:
        sampler = clip["samplers"][channel["sampler"]]
        keys = values(sampler["input"])
        samples = values(sampler["output"])
        size = len(samples) // len(keys)
        start = keys.index(time) * size
        target = channel["target"]
        pose.setdefault(target["node"], {})[target["path"]] = samples[start:start+size]
    return pose


@lru_cache(maxsize=3)
def child_seat(profile):
    """Bake the same D02 seat with an authored pose for the catalogue layout."""
    source = ROOT / "models/double_cabin/D02_w550_state1_wheel_M.glb"
    doc, binary = read_glb(source)
    nodes = doc["nodes"]
    root = next(i for i, node in enumerate(nodes) if node.get("name") == "D02_Front_Shell")
    pose = {}
    if profile == "sitting":
        pose = seat_keyframe(doc, binary, "D02_Front_canopy", 0.25)
    elif profile == "tucked":
        pose = seat_keyframe(doc, binary, "D02_Front_legrest", 1.0)
    parts = {}

    def add(index, parent_pos, parent_quat):
        node = nodes[index]
        override = pose.get(index, {})
        offset = rotate_vec(parent_quat, node.get("translation", (0, 0, 0)))
        position = tuple(a + b for a, b in zip(parent_pos, offset))
        local_rotation = override.get("rotation", node.get("rotation", (0, 0, 0, 1)))
        quaternion = compose_rotations(parent_quat, local_rotation)
        if "mesh" in node:
            for primitive in doc["meshes"][node["mesh"]]["primitives"]:
                material = doc["materials"][primitive["material"]]["name"].removeprefix("D02_")
                if material not in parts:
                    parts.update(group(material))
                g = parts[material]
                attributes = primitive["attributes"]
                points = read_positions(doc, binary, attributes["POSITION"])
                weights = override.get("weights", ())
                for weight, target in zip(weights, primitive.get("targets", ())):
                    if weight:
                        offsets = read_positions(doc, binary, target["POSITION"])
                        points = [tuple(a + weight*b for a, b in zip(p, delta)) for p, delta in zip(points, offsets)]
                points = [tuple(a + b for a, b in zip(rotate_vec(quaternion, p), position)) for p in points]
                if any(weights):
                    for a, b, c in zip(points[::3], points[1::3], points[2::3]):
                        g.triangle(a, b, c)
                else:
                    g.vertices.extend(points)
                    g.normals.extend(rotate_vec(quaternion, n) for n in read_positions(doc, binary, attributes["NORMAL"]))
        for child in node.get("children", ()):
            add(child, position, quaternion)

    add(root, (0, 0, 0), (0, 0, 0, 1))
    return parts


@lru_cache(maxsize=None)
def cabin_geometry(cabin_id, width, profile="standard"):
    low = profile == "low"
    if cabin_id in ("S01", "S08", "S09"):
        from single_cabin_catalog import special_cabin_geometry
        return special_cabin_geometry(cabin_id, width, profile)
    if cabin_id in ("S02", "S03", "S04"):
        return pet_cabin(cabin_id, low)
    if cabin_id == "S05":
        return child_seat(profile)
    if cabin_id == "S06":
        return bassinet(low)
    if cabin_id == "S07":
        return infant_carrier(low)
    if cabin_id in ("S10", "S11", "S12"):
        from multi_cabin_geometry import multi_cabin_geometry
        return multi_cabin_geometry(cabin_id, width, profile)
    if cabin_id in ("S13", "S14", "S15"):
        return cargo_cabin(cabin_id, paired=profile == "paired")
    raise ValueError(f"Unknown cabin {cabin_id}")
