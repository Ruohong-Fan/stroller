"""Read the V46 fold poses and sample their articulated frame links in between."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math

from catalog_frame import prepare_catalog_frame
from create_d02_double_seat import (
    AXIS_Z, D02_RISE, FRONT_PORT_ADVANCE, FRONT_PORT_DROP, HIGH_PORT_RISE, HIGH_PORT_SETBACK, rotation,
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
    doc, binary = _native(width, wheel, state)
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


def _sample_pose(opening, motion, phase, height):
    doc, binary, installed = opening
    pivots, frames, tracks = motion
    first = int(phase)-1
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
    _ground(sample, ground)
    datums = {}
    for station, datum in installed.items():
        x, y, z = _map_xy(unlevel(datum.center), transforms[2 if station == "L" else 3])
        datums[station] = FoldDatum((x, y-ground, z), angles[2 if station == "L" else 3])
    return sample, binary, datums


def prepare_fold_samples(width, wheel, ports=(), *, steps=8):
    """Return exact poses and a sampled linkage approximation across both stages.

    Phases 1, 2 and 3 are the authored GLBs. Other phases follow the inferred
    articulated link trajectory for geometric screening between those poses.
    """
    if steps < 1:
        raise ValueError("Each folding stage needs at least one sample")
    opening = prepare_fold_frame(width, wheel, 1, ports)
    rest, rest_bin = _native(width, wheel, 1)
    native_l = _center(_points(rest, rest_bin, _hosts(rest)["L", 1]))
    dx, dy = (-FRONT_PORT_ADVANCE, -FRONT_PORT_DROP) if width != 380 and tuple(ports) == ("L", "H") else (0, 0)
    _, expected_y = _rot(native_l[0]+dx, native_l[1]+dy, opening[2]["L"].angle)
    height = expected_y-opening[2]["L"].center[1]
    motion = _motion(width)
    result = [(1.0, *opening)]
    for first in (1, 2):
        for i in range(1, steps):
            phase = first+i/steps
            result.append((phase, *_sample_pose(opening, motion, phase, height)))
        result.append((float(first+1), *prepare_fold_frame(width, wheel, first+1, ports)))
    return result
