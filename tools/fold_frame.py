"""Read the V46 fold poses and sample their articulated frame links in between."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math
import struct

from catalog_frame import prepare_catalog_frame
from create_d02_double_seat import (
    AXIS_Z, D02_RISE, FRONT_PORT_ADVANCE, FRONT_PORT_DROP, Geometry, HIGH_PORT_RISE, HIGH_PORT_SETBACK,
    accessor, cross, rotation, sub,
)
from d02_front_mount import relocate_front_ports, relocate_high_ports
from generate_wheel_variants import DEFAULT_WHEEL, MODEL_DIR, compact_active_scene, read_glb
from universal_mount import _center, _fold_rotation, _hosts, _points, remove_receiver_nodes, standardize_receivers


@dataclass(frozen=True)
class FoldDatum:
    center: tuple[float, float, float]
    angle: float  # Radians about the GLB Z axis, from the native open attitude.


def _rot(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return x*c-y*s, x*s+y*c


def _interpolate(a, b, t):
    return (1-t)*a + t*b


def _code(node):
    return node["name"].split(".", 1)[0]


def _leg_triangle_indices(doc, binary, primitive, count):
    if "indices" not in primitive:
        return range(count)
    field = doc["accessors"][primitive["indices"]]
    view = doc["bufferViews"][field["bufferView"]]
    code = {5121: "B", 5123: "H", 5125: "I"}[field["componentType"]]
    offset = view.get("byteOffset", 0) + field.get("byteOffset", 0)
    return struct.unpack_from(f"<{field['count']}{code}", binary, offset)


def _clip_leg_face(polygon, progress, plane, side):
    section = []
    previous = polygon[-1]
    distance = progress(previous) - plane
    for vertex in polygon:
        next_distance = progress(vertex) - plane
        if (distance*side >= 0) != (next_distance*side >= 0):
            scale = distance/(distance-next_distance)
            section.append(tuple(_interpolate(a, b, scale) for a, b in zip(previous, vertex)))
        if next_distance*side >= 0:
            section.append(vertex)
        previous, distance = vertex, next_distance
    return section


def _rear_leg_relief(points, indices):
    """Flatten the wheel-side wall at the lower pivot, then blend into the tube."""
    mean_x, mean_y, mean_z = _mean(points)
    xx = sum((x-mean_x)**2 for x, _, _ in points)
    yy = sum((y-mean_y)**2 for _, y, _ in points)
    xy = sum((x-mean_x)*(y-mean_y) for x, y, _ in points)
    angle = .5*math.atan2(2*xy, xx-yy)
    ax, ay = math.cos(angle), math.sin(angle)
    if ay < 0:
        ax, ay = -ax, -ay
    along = [x*ax+y*ay for x, y, _ in points]
    lower, upper = min(along), max(along)
    side = math.copysign(1, mean_z)
    inner = min(side*p[2] for p in points)
    depth = .006
    ends = (.12, .145, .17, .195, .22, .25)

    def progress(point):
        return (point[0]*ax+point[1]*ay-lower)/(upper-lower)

    def trimmed(point):
        x, y, z = point
        u = min(1., max(0., (progress(point)-ends[0])/(ends[-1]-ends[0])))
        limit = inner + depth*(1.-u*u*(3.-2.*u))
        return x, y, side*max(side*z, limit)

    shape = Geometry()

    def add_face(a, b, c):
        if math.hypot(*cross(sub(b, a), sub(c, a))) > 1e-11:
            shape.triangle(a, b, c)

    for i in range(0, len(indices), 3):
        face = [points[index] for index in indices[i:i+3]]
        if min(side*p[2] for p in face) >= inner+depth or min(map(progress, face)) >= ends[-1]:
            add_face(*face)
            continue
        sections = []
        for plane in ends:
            if min(map(progress, face)) < plane-1e-7 and max(map(progress, face)) > plane+1e-7:
                sections.append(_clip_leg_face(face, progress, plane, -1.))
                face = _clip_leg_face(face, progress, plane, 1.)
        sections.append(face)
        for section in sections:
            a, *edges = [trimmed(p) for p in section]
            for b, c in zip(edges, edges[1:]):
                add_face(a, b, c)
    return shape


def relieve_rear_legs(doc, binary):
    """Add the large-wheel relief to the two V46 rear tubes before animating them.

    Accepts the published frame's raw TRS or already baked assembly coordinates.
    Each pin axis and upper attachment stay where they were. Only the inboard
    wall of the lower tube is reshaped, ahead of the transverse saddle.
    """
    legs = [i for i, node in enumerate(doc["nodes"]) if _code(node) in ("V46_rear_leg_1", "V46_rear_leg_-1")]
    if len(legs) != 2:
        raise ValueError("The folding chassis needs both V46 rear tubes")
    animated = {c["target"].get("node") for a in doc.get("animations", ()) for c in a["channels"]}
    for i in legs:
        if i in animated:
            raise ValueError("Apply rear tube relief before exporting its folding animation")
        node = doc["nodes"][i]
        source = doc["meshes"][node["mesh"]]
        primitive, = source["primitives"]
        points = _points(doc, binary, node)
        indices = _leg_triangle_indices(doc, binary, primitive, len(points))
        shape = _rear_leg_relief(points, indices)
        updated = {**primitive, "attributes": {
            "POSITION": accessor(doc, binary, shape.vertices, "VEC3", vertex=True, bounds=True),
            "NORMAL": accessor(doc, binary, shape.normals, "VEC3", vertex=True),
        }}
        updated.pop("indices", None)
        for field in ("matrix", "translation", "rotation", "scale"):
            node.pop(field, None)
        node["mesh"] = len(doc["meshes"])
        doc["meshes"].append({**source, "primitives": [updated]})
    doc["buffers"][0]["byteLength"] = len(binary)


def _native(width, wheel, state):
    doc, binary = read_glb(MODEL_DIR / f"w{width}_state{state}_wheel_{wheel}.glb")
    remove_receiver_nodes(doc)
    return compact_active_scene(doc, binary)


def _center_at(centers, station, ground, angle=0, parent_angle=0):
    a, b = centers[station, -1], centers[station, 1]
    x, y = _rot((a[0]+b[0])/2, (a[1]+b[1])/2, parent_angle)
    return FoldDatum((x, y-ground, (a[2]+b[2])/2), angle+parent_angle)


def _height(doc, binary):
    return min(point[1] for node in doc["nodes"]
               if _code(node).startswith(("FL_tire", "FR_tire", "RL_tire", "RR_tire", "ground_pad"))
               for point in _points(doc, binary, node))


def _ground(doc, height):
    # Published narrow meshes store object TRS, while the new receiver meshes
    # are already baked to the world. Inserting one root keeps both intact.
    if abs(height) > 1e-7:
        scene = doc["scenes"][doc.get("scene", 0)]
        roots = scene["nodes"]
        scene["nodes"] = [len(doc["nodes"])]
        doc["nodes"].append({"name": "V46_Ground_Datum", "translation": [0, -height, 0], "children": roots})


def _move_sockets(doc, binary, angles):
    dx, dy = _rot(-FRONT_PORT_ADVANCE, -FRONT_PORT_DROP, angles["L"])
    relocate_front_ports(doc, binary, forward=-dx, down=-dy)
    dx, dy = _rot(HIGH_PORT_SETBACK, HIGH_PORT_RISE, angles["H"])
    relocate_high_ports(doc, binary, aft=dx, up=dy)


def prepare_fold_frame(width, wheel, state, ports=()):
    """Build one of the three authored postures and its mounting transforms."""
    if state == 1:
        doc, binary, centers, (angle, ground) = prepare_catalog_frame(width, wheel, ports)
        return doc, binary, {station: _center_at(centers, station, ground, parent_angle=angle) for station in "LMH"}
    if state not in (2, 3):
        raise ValueError(f"Not a folding posture: {state}")

    rest, rest_bin = _native(width, wheel, 1)
    doc, binary = _native(width, wheel, int(state))
    hinges = _hosts(doc)
    reference = _hosts(rest)
    angles = {
        station: _fold_rotation(_points(doc, binary, hinges[station, 1]),
                                _points(rest, rest_bin, reference[station, 1]))
        for station in "LMH"
    }
    if width != 380 and tuple(ports) == ("L", "H"):
        _move_sockets(doc, binary, angles)
    centers = standardize_receivers(doc, binary, reference=(rest, rest_bin), rise=D02_RISE)
    height = _height(doc, binary)
    _ground(doc, height)
    return doc, binary, {station: _center_at(centers, station, height, angles[station]) for station in "LMH"}


# Each part moves with a rigid frame segment. These named points are physically
# the rear axle, the lower/upper rear-leg joints, and the main folding pin.
_PIVOTS = ("RL_tire", "horizontal_rear_receiver_1", "V46_cross_main_insert_1", "V46_flush_fold_pin_1")
_LINKS = ("fixed_rear_module_root_1", "V46_rear_leg_1", "main_lower_1", "main_upper_1")


def _transforms(pivots, angles, *, targets=None):
    """Forward kinematics from grounded rear axle to upper tube."""
    result = []
    for i, (anchor, angle) in enumerate(zip(pivots, angles)):
        if targets is not None:
            x, y = targets[i][:2]
        elif i == 0:
            x, y = anchor[:2]
        else:
            a, b, c = result[-1]
            x, y = _rot(*anchor[:2], c)
            x, y = x+a, y+b
        cx, cy = _rot(*anchor[:2], angle)
        result.append((x-cx, y-cy, angle))
    return result


def _map_xy(pos, transform):
    dx, dy, angle = transform
    x, y = _rot(*pos[:2], angle)
    return (x+dx, y+dy, pos[2])


def _mean(points):
    return tuple(sum(axis)/len(points) for axis in zip(*points))


@cache
def _motion(width):
    """Recover linkage angles and moving-member offsets from corresponding CAD parts."""
    native = [_native(width, DEFAULT_WHEEL[width], state) for state in (1, 2, 3)]
    points = [{_code(n): _points(d, b, n) for n in d["nodes"]} for d, b in native]
    pivot_tracks = [tuple(_center(geometry[name]) for name in _PIVOTS) for geometry in points]
    pivots = pivot_tracks[0]
    angles = [tuple(_fold_rotation(geometry[name], points[0][name]) for name in _LINKS)
              for geometry in points]
    exact = [_transforms(pivots, a, targets=p) for a, p in zip(angles, pivot_tracks)]
    tracks = {}
    for name, start in points[0].items():
        base_center = _mean(start)
        endpoints = [geometry[name] for geometry in points]
        centers = [_mean(finish) for finish in endpoints]
        rotations = [_fold_rotation(finish, start) if len(finish) == len(start) else None for finish in endpoints]

        def error(link):
            total = 0.0
            for i in (1, 2):
                guess = _map_xy(base_center, exact[i][link])
                total += math.dist(centers[i], guess)
                if rotations[i] is not None:
                    total += .08 * abs(rotations[i]-angles[i][link])
            return total

        if name.startswith(("handle_slider", "ground_pad", "round_handle")):
            # The handle telescopes along the upper tube during the fold.
            link = 3
        else:
            link = min(range(4), key=error)
        # Residual movement is measured in the moving link's coordinates so
        # a telescoping part remains on its rail between the native poses.
        residual = []
        for i, center in enumerate(centers):
            guess = _map_xy(base_center, exact[i][link])
            dx, dy = _rot(center[0]-guess[0], center[1]-guess[1], -angles[i][link])
            angle = rotations[i] if rotations[i] is not None else angles[i][link]
            residual.append((dx, dy, center[2]-guess[2], angle-angles[i][link]))
        tracks[name] = (link, base_center, residual)
    return pivots, angles, tracks


def _attachment_link(name):
    if name.startswith("Universal_Cabin_Receiver_"):
        return 2 if name.split("_")[-2] == "L" else 3
    if name.startswith("D02_FrontLowerReceiverGusset_"):
        return 2
    if name.startswith("D02_RearUpperReceiverGusset_"):
        return 3
    return None


def _sample_pose(opening, motion, phase, height, *, ground_inline=False):
    doc, binary, installed = opening
    pivots, frames, tracks = motion
    first = min(int(phase)-1, len(frames)-2)
    t = phase-(first+1)
    angles = tuple(_interpolate(a, b, t) for a, b in zip(frames[first], frames[first+1]))
    transforms = _transforms(pivots, angles)
    level_angle = installed["L"].angle

    def unlevel(p):
        # An installed datum and its corresponding native receiver differ by
        # the one rotation and one vertical offset used by prepare_catalog_frame.
        return (*_rot(p[0], p[1]+height, -level_angle), p[2])

    nodes = []
    for node in doc["nodes"]:
        name = _code(node)
        link, center, residuals = tracks.get(name, (_attachment_link(name), (0, 0, 0), [(0, 0, 0, 0)]*3))
        if link is None:
            raise ValueError(f"No fold linkage for {name}")
        dx, dy, dz, turn = (_interpolate(a, b, t) for a, b in zip(residuals[first], residuals[first+1]))
        rx, ry = _rot(dx, dy, angles[link])
        cx, cy, cz = _map_xy(center, transforms[link])
        angle = angles[link]+turn
        ox, oy = _rot(*center[:2], angle)
        lx, ly = _rot(0, height, angle-level_angle)
        nodes.append({**node, "translation": [cx+rx-ox+lx, cy+ry-oy+ly, cz+dz-center[2]],
                      "rotation": rotation(AXIS_Z, math.degrees(angle-level_angle))})

    # Ground offsets are inferred from the running-surface geometry, then the
    # same offset is applied to both chassis and all receiver datums.
    sample = {**doc, "nodes": nodes, "scenes": [dict(scene) for scene in doc["scenes"]]}
    ground = _height(sample, binary)
    if ground_inline:
        for node in nodes:
            node["translation"][1] -= ground
    else:
        _ground(sample, ground)
    datums = {}
    for station, datum in installed.items():
        x, y, z = _map_xy(unlevel(datum.center), transforms[2 if station == "L" else 3])
        datums[station] = FoldDatum((x, y-ground, z), angles[2 if station == "L" else 3])
    return sample, binary, datums


def _opening(width, wheel, ports):
    opening = prepare_fold_frame(width, wheel, 1, ports)
    rest, rest_bin = _native(width, wheel, 1)
    native_l = _center(_points(rest, rest_bin, _hosts(rest)["L", 1]))
    dx, dy = (-FRONT_PORT_ADVANCE, -FRONT_PORT_DROP) if width != 380 and tuple(ports) == ("L", "H") else (0, 0)
    _, expected_y = _rot(native_l[0]+dx, native_l[1]+dy, opening[2]["L"].angle)
    return opening, expected_y-opening[2]["L"].center[1]


def prepare_fold_samples(width, wheel, ports=(), *, steps=8):
    """Return exact poses and a sampled linkage approximation across both stages.

    Phases 1, 2 and 3 are the authored GLBs. Other phases follow the inferred
    articulated link trajectory for geometric screening between those poses.
    """
    if steps < 1:
        raise ValueError("Each folding stage needs at least one sample")
    opening, height = _opening(width, wheel, ports)
    motion = _motion(width)
    result = [(1.0, *opening)]
    for first in (1, 2):
        for i in range(1, steps):
            phase = first+i/steps
            result.append((phase, *_sample_pose(opening, motion, phase, height)))
        result.append((float(first+1), *prepare_fold_frame(width, wheel, first+1, ports)))
    return result


def prepare_continuous_fold(width, wheel, ports=(), *, steps=32, end_phase=3.0):
    """Sample both fold segments on one opening mesh for rigid glTF animation.

    All phases share the opening mesh buffers and node indices. The returned
    phases are evenly spaced from 1 to the selected mechanical stop, with the
    opening wheel and fork geometry carried through the entire motion.
    """
    if steps < 1:
        raise ValueError("Each folding stage needs at least one sample")
    if not 1.0 < end_phase <= 3.0:
        raise ValueError("The folding stop must follow the open pose and stay within the linkage path")
    opening, height = _opening(width, wheel, ports)
    if wheel == "L" and end_phase > 2.45:
        relieve_rear_legs(opening[0], opening[1])
    motion = _motion(width)
    result = [(1.0, *opening)]
    for i in range(1, 2*steps+1):
        phase = end_phase if i == 2*steps else _interpolate(1., end_phase, i/(2*steps))
        result.append((phase, *_sample_pose(opening, motion, phase, height, ground_inline=True)))
    return result
