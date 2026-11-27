#!/usr/bin/env python3
"""Export the captured-slide folding path shared by the preview and the mesh checks."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from functools import cache, lru_cache
import json
import math
from pathlib import Path

import numpy as np

from create_d02_double_seat import AXIS_Z, D02_LAND_Y, Geometry, MATERIALS, accessor, mesh_node, rotation
from create_folded_catalog import (
    append_module, cabin_modules, catalogue, module_layout, move_point, station_matrix,
)
from fold_clearance import MeshPart, first_contact, ground_collision, node_matrix, pose_parts, scene_parts
from fold_frame import prepare_continuous_fold
from generate_wheel_variants import ROOT, SPECS, write_glb


GUIDE_RADIUS = 0.0085
DEFAULT_PROFILES = Path(__file__).with_name("attached_fold_profiles.json")
MATERIALS["captive_guide"] = ([0.075, 0.395, 0.452, 1.0], .58, .37)
MATERIALS["captive_shaft"] = ([0.46, 0.64, 0.68, 1.0], .72, .31)


@dataclass(frozen=True)
class Travel:
    x: float = 0
    y: float = 0
    angle: float = 0
    turn_by: float = 1
    move_by: float = 1


def easing(progress, ending):
    t = min(1., progress / ending)
    return t * t * (3. - 2. * t)


def motion_matrix(reference_inverse, frame, travel, progress):
    turn = math.radians(travel.angle * easing(progress, travel.turn_by))
    shift = easing(progress, travel.move_by)
    c, s = math.cos(turn), math.sin(turn)
    slide = np.array([[c, -s, 0, travel.x*shift], [s, c, 0, travel.y*shift],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    return frame @ slide @ reference_inverse


def read_profiles(path):
    return json.loads(path.read_text()) if path.exists() else {}


def design_profiles(profiles, identity, width, wheel, count):
    if identity == "D02":
        return [Travel(-.015, .020), Travel(-.080, 0, 20)], None
    candidates = (f"{identity}/{width}/{wheel}", f"{identity}/{width}", identity)
    plan = next((profiles[key] for key in candidates if key in profiles), None)
    if plan is None:
        raise ValueError(f"No attached folding plan for {identity} {width}/{wheel}")
    legs = plan["travel"] if isinstance(plan, dict) else plan
    shared = plan.get("shared") if isinstance(plan, dict) else None
    if len(legs) != count:
        raise ValueError(f"Expected {count} travel profiles for {identity}: {plan!r}")
    return [Travel(*leg) for leg in legs], shared


def attached_stop(width, wheel):
    # The retained-cabin stop engages before the bare-frame storage latch.
    # Narrow and medium frames with the large tire need more rear-joint space.
    if wheel == "L":
        return 2.43 if width < 650 else 2.91
    return 2.86


@lru_cache(maxsize=2)
def chassis_motion(width, wheel, ports, steps):
    samples = prepare_continuous_fold(width, wheel, ports, steps=steps, end_phase=attached_stop(width, wheel))
    rest_doc = samples[0][1]
    spans = {n["extras"]["station"]: abs(n["extras"]["nativeLand"][2])
             for n in rest_doc["nodes"] if n.get("name", "").startswith("Universal_Cabin_Receiver_")}
    frames = []
    for phase, doc, binary, datums in samples:
        progress = (phase - 1.) / 2.
        parts = scene_parts(doc, binary)
        wheel_parts = [part for part in parts if part.name.startswith(("FL_", "FR_", "RL_tire", "RR_tire"))]
        for member in wheel_parts:
            # Ignore components of the same wheel assembly and its steering fit.
            corner = member.name.split("_")[0]
            excluded = "vertical_front_guide_" + ("-1" if corner == "FL" else "1") if "steering_bearing" in member.name else None
            other_hardware = [part for part in parts if not part.name.startswith(corner + "_")
                              and part.name.split(".")[0] != excluded]
            if hit := first_contact([member], other_hardware):
                raise ValueError(f"{width}/{wheel} at {progress:.4f}: {member.name} contacts {hit[1].name}")
        rearbar = [part for part in parts if part.name.split(".")[0] == "V46_rearbar"]
        lower_rails = [part for part in parts if part.name.startswith("main_lower_")]
        if hit := first_contact(rearbar, lower_rails):
            raise ValueError(f"{width}/{wheel} at {progress:.4f}: {hit[0].name} contacts {hit[1].name}")
        frames.append((progress, doc, binary, datums, parts))
    return spans, frames


def links_for(module, opened, datums, transform, spans):
    joints = []
    for station in module.stations:
        original = station_matrix(opened[station])
        carrier = station_matrix(datums[station])
        for side in (-1, 1):
            land = np.array((0., D02_LAND_Y, side*spans[station]))
            start = move_point(carrier, land)
            stop = move_point(transform, move_point(original, land))
            joints.append((station, start, stop))
    return joints


def link_pose(start, stop):
    vector = stop - start
    length = math.hypot(*vector)
    angle = math.degrees(math.atan2(-vector[0], vector[1])) if length > 1e-8 else 0.
    return start.tolist(), rotation(AXIS_Z, angle), [1., length, 1.]


@cache
def guide_shape():
    shaft = Geometry()
    shaft.beam((0., 0., 0.), (0., 1., 0.), GUIDE_RADIUS, sides=12)
    return shaft


def guide_part(name, station, start, stop):
    position, attitude, scale = link_pose(start, stop)
    matrix = node_matrix({"translation": position, "rotation": attitude, "scale": scale})
    triangles = np.array(guide_shape().vertices).reshape(-1, 3, 3)
    triangles = triangles @ matrix[:3, :3].T + matrix[:3, 3]
    return MeshPart((name,), "D02_mounts", triangles, station)


def check_sample(spec, progress, frame, placed, guides):
    link_meshes = [
        [guide_part(f"Guide{i+1}_{j+1}", station, start, stop)
         for j, (station, start, stop) in enumerate(connections) if math.dist(start, stop) >= .003]
        for i, connections in enumerate(guides)
    ]
    for i, parts in enumerate(placed):
        others = [part for index, items in enumerate(placed) if index != i for part in items]
        if (hit := ground_collision(str(i), parts, progress, tolerance=0.0005)):
            raise ValueError(f"{spec} at {progress:.4f}: {hit.source} projects below the standing plane")
        if (hit := first_contact(parts, frame, skip_mating=True)) or (hit := first_contact(parts, others)):
            a, b, _ = hit
            raise ValueError(f"{spec} at {progress:.4f}: {a.name} contacts {b.name}")
        moving = link_meshes[i]
        other_guides = [guide for index, items in enumerate(link_meshes) if index != i for guide in items]
        # Contact at the bracket which the guide plugs into is intentional.
        if (hit := first_contact(moving, frame, skip_mating=True)) or (hit := first_contact(moving, others + other_guides)):
            a, b, _ = hit
            raise ValueError(f"{spec} at {progress:.4f}: {a.name} contacts {b.name}")


def write_animation(doc, binary, nodes, times):
    time_id = accessor(doc, binary, [(p,) for p in times], "SCALAR", bounds=True)
    samplers, channels = [], []
    for index, properties in nodes:
        for path, values in properties.items():
            if all(np.allclose(v, values[0], rtol=0, atol=1e-7) for v in values[1:]):
                continue
            value_id = accessor(doc, binary, values, "VEC4" if path == "rotation" else "VEC3")
            samplers.append({"input": time_id, "output": value_id, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers)-1, "target": {"node": index, "path": path}})
    doc["animations"] = [{"name": "STROLLER_FOLD", "samplers": samplers, "channels": channels}]


def add_motion_node(doc, name, children):
    index = len(doc["nodes"])
    doc["nodes"].append({"name": name, "children": children})
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(index)
    return index


def rigid_pose(matrix):
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return matrix[:3, 3].tolist(), rotation(AXIS_Z, angle)


def add_cabins(doc, binary, modules, transforms, active_frame, animated):
    scene = doc["scenes"][doc.get("scene", 0)]
    tracks = []
    for module, matrices in zip(modules, transforms):
        before = len(scene["nodes"])
        append_module(doc, binary, module, np.eye(4))
        children = scene["nodes"][before:]
        del scene["nodes"][before:]
        index = add_motion_node(doc, module.key + "_CapturedCarrier", children)
        translations, rotations = zip(*(rigid_pose(matrix) for matrix in matrices))
        doc["nodes"][index].update(translation=translations[active_frame], rotation=rotations[active_frame])
        if animated:
            tracks.append((index, {"translation": translations, "rotation": rotations}))
    return tracks


def add_guides(doc, binary, joint_paths, active_frame, animated):
    jacket, shaft = Geometry(), Geometry()
    jacket.beam((0., 0., 0.), (0., .56, 0.), GUIDE_RADIUS, sides=12)
    shaft.beam((0., .49, 0.), (0., 1., 0.), GUIDE_RADIUS * .73, sides=12)
    scene = doc["scenes"][doc.get("scene", 0)]
    tracks, mesh_id = [], None
    for cabin, joint_series in enumerate(joint_paths):
        for joint, path in enumerate(zip(*joint_series)):
            if all(math.dist(start, stop) < .001 for _, start, stop in path):
                continue
            poses = [link_pose(start, stop) for _, start, stop in path]
            shifts, turns, scales = zip(*poses)
            name = f"CapturedGuide_{cabin+1}_{joint+1}"
            if mesh_id is None:
                index = mesh_node(doc, binary, name, {"captive_guide": jacket, "captive_shaft": shaft})
                mesh_id = doc["nodes"][index]["mesh"]
            else:
                index = len(doc["nodes"])
                doc["nodes"].append({"name": name, "mesh": mesh_id})
            scene["nodes"].append(index)
            doc["nodes"][index].update(translation=shifts[active_frame], rotation=turns[active_frame], scale=scales[active_frame])
            if animated:
                tracks.append((index, {"translation": shifts, "rotation": turns, "scale": scales}))
    return tracks


def scan_path(label, modules, profiles, spans, frames, shared_site=None):
    opened = frames[0][3]
    inv = {site: np.linalg.inv(station_matrix(datum)) for site, datum in opened.items()}
    transforms = [[] for _ in modules]
    links = [[] for _ in modules]
    for progress, _, _, datums, body in frames:
        placed, exposed = [], []
        for module, travel, matrices, link_series in zip(modules, profiles, transforms, links):
            site = shared_site or module.stations[0]
            placement = motion_matrix(inv[site], station_matrix(datums[site]), travel, progress)
            joints = links_for(module, opened, datums, placement, spans)
            matrices.append(placement)
            link_series.append(joints)
            placed.append(pose_parts(module.parts, placement))
            exposed.append(joints)
        check_sample(label, progress, body, placed, exposed)
    return transforms, links


def build_variant(output, mode, spec, names, width, wheel, profiles, steps):
    identity = spec["id"]
    label = f"{identity}/{width}/{wheel}"
    ports = ("L", "H") if width != 380 and spec["ports"] == ["LOW", "HIGH"] else ()
    folder = output / f"{mode}_cabin"
    source = ROOT / "models" / f"{mode}_cabin" / f"{identity}_w{width}_state1_wheel_{wheel}.glb"
    modules = cabin_modules(source, module_layout(mode, spec, names))
    travel, shared_site = design_profiles(profiles, identity, width, wheel, len(modules))
    spans, frames = chassis_motion(width, wheel, ports, steps)
    transforms, joint_paths = scan_path(label, modules, travel, spans, frames, shared_site)
    finish = frames[-1][0]
    times = [data[0] / finish for data in frames]

    note = "座舱全程与车架相连。"
    if shared_site:
        note += "两舱随联动托座一起转动，让出收折空间。"
    elif any(travel_step != Travel() for travel_step in travel):
        note += "侧导座随车架滑移或转动，让出收折空间。"
    if identity == "D02":
        note += "两舱同向，靠背 95°、顶篷第 1 档、腿托收起。"
    if "S15" in (identity, spec.get("lower"), spec.get("upper")):
        note += "软载货舱按空舱检查。"
    if wheel == "L" and width == 650:
        note += "后支撑梁内侧留出前轮通过的空间。"
    elif wheel == "L":
        note += "带舱限位提前结束收折，为大轮和后支撑梁留出空间。"
    note += "动画终点为带舱收折限位。"
    summary = {
        "retained": [{"name": module.name} for module in modules], "removed": [],
        "note": note, "sampleCount": len(frames), "initialProgress": 0.5, "hasMarker": False,
    }
    folder.mkdir(parents=True, exist_ok=True)
    for posture, frame_index in ((2, len(frames)//2), (3, len(frames)-1)):
        animated = posture == 2
        _, template, reference, _, _ = frames[frame_index]
        doc, binary = deepcopy(template), bytearray(reference)
        tracks = []
        if animated:
            for i, node in enumerate(doc["nodes"]):
                poses = [data[1]["nodes"][i] for data in frames]
                tracks.append((i, {
                    "translation": [pose.get("translation", [0, 0, 0]) for pose in poses],
                    "rotation": [pose.get("rotation", [0, 0, 0, 1]) for pose in poses],
                }))
        tracks.extend(add_cabins(doc, binary, modules, transforms, frame_index, animated))
        tracks.extend(add_guides(doc, binary, joint_paths, frame_index, animated))
        if animated:
            write_animation(doc, binary, tracks, times)
        doc["scenes"][doc.get("scene", 0)].setdefault("extras", {})["strollerFold"] = summary
        doc["buffers"][0]["byteLength"] = len(binary)
        write_glb(folder / f"{identity}_w{width}_state{posture}_wheel_{wheel}.glb", doc, binary)
    print(f"{label}: {len(modules)} attached, {len(frames)} checked poses", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--cabin", action="append")
    parser.add_argument("--width", action="append", type=int, choices=(380, 550, 650))
    parser.add_argument("--wheel", action="append", choices=tuple(SPECS))
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    profiles = read_profiles(args.profiles)
    jobs = []
    for mode, spec, names in catalogue():
        if args.cabin and spec["id"] not in args.cabin:
            continue
        for width in spec["widths"]:
            if args.width and width not in args.width:
                continue
            for wheel in args.wheel or SPECS:
                jobs.append((width, wheel, tuple(spec["ports"]), mode, spec, names))
    for width, wheel, _, mode, spec, names in sorted(jobs, key=lambda job: job[:3]):
        build_variant(args.output, mode, spec, names, width, wheel, profiles, args.steps)


if __name__ == "__main__":
    main()
