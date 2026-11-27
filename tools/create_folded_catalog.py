#!/usr/bin/env python3
"""Export the checked intermediate and standing fold poses of each cabin assembly."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re

import numpy as np

from catalog_cabin_geometry import seat_keyframe
from create_cabin_catalog import load_catalog
from create_d02_double_seat import Geometry, MATERIALS, ROOT, accessor, mesh_node, unit
from fold_clearance import (
    Collision, FoldSample, choose_fold_order, ground_collision, node_matrix, pose_parts, scene_parts,
)
from fold_frame import prepare_fold_samples
from generate_wheel_variants import SPECS, compact_active_scene, read_glb, read_positions, write_glb


STAGES = (2, 3)
MATERIALS["fold_contact"] = ([1.0, .30, .025, 1], 0, .48)


@dataclass(frozen=True)
class CabinModule:
    key: str
    name: str
    cabin_id: str
    stations: tuple[str, ...]
    doc: dict
    binary: bytearray
    parts: list


def catalogue():
    index = ROOT / "index.html"
    names = dict(re.findall(r"\{ id: '(S\d\d)', name: '([^']+)'", index.read_text()))
    for mode, spec, _ in load_catalog(index):
        yield mode, spec, names
    yield "double", {"id": "D02", "lower": "S05", "upper": "S05",
                     "widths": [550, 650], "ports": ["LOW", "HIGH"]}, names


def module_layout(mode, spec, names):
    cabin_id = spec["id"]
    if mode == "single":
        part = cabin_id + "_Cabin"
        if cabin_id == "S01":
            return [(part, names[cabin_id], cabin_id, ("H", "M"))]
        if cabin_id in ("S08", "S09"):
            return [(part, names[cabin_id], cabin_id, ("M", "H")),
                    (part + "_Footrest", "成人舱脚踏", "", ("L",))]
        return [(part, names[cabin_id], cabin_id, ("L" if cabin_id == "S15" else "M",))]

    lower, upper = spec["lower"], spec["upper"]
    front, rear = ("D02_Front", "D02_Rear") if cabin_id == "D02" else (cabin_id + "_Lower", cabin_id + "_Upper")
    low = spec["ports"][0][0]
    high = ("H", "M") if upper == "S01" else (spec["ports"][1][0],)
    return [(front, "下舱 · " + names[lower], lower, (low,)),
            (rear, "上舱 · " + names[upper], upper, high)]


def apply_morph(doc, binary, index, weights):
    mesh = deepcopy(doc["meshes"][index])
    mesh.pop("weights", None)
    for primitive in mesh["primitives"]:
        attributes = primitive["attributes"]
        targets = primitive.pop("targets", ())
        for attr in ("POSITION", "NORMAL"):
            selected = [(weight, target[attr]) for weight, target in zip(weights, targets)
                        if weight and attr in target]
            if not selected:
                continue
            values = np.asarray(read_positions(doc, binary, attributes[attr]))
            for weight, target in selected:
                values += weight * np.asarray(read_positions(doc, binary, target))
            if attr == "NORMAL":
                values = [unit(v) for v in values]
            attributes[attr] = accessor(doc, binary, values, "VEC3", vertex=True, bounds=attr == "POSITION")
    doc["meshes"].append(mesh)
    return len(doc["meshes"]) - 1


def set_matrix(node, matrix):
    for key in ("translation", "rotation", "scale"):
        node.pop(key, None)
    node["matrix"] = matrix.flatten(order="F").tolist()


def cabin_modules(path, layout, *, rear_reversed=False):
    source, source_binary = read_glb(path)
    scene = source["scenes"][source.get("scene", 0)]
    root_ids = list(scene["nodes"])
    result = []
    for key, name, cabin_id, stations in layout:
        excluded = tuple(other for other, _, _, _ in layout if other.startswith(key + "_"))
        adapter = f"D02_WideAdapter_{stations[0]}_"
        roots = []
        for index in root_ids:
            node_name = source["nodes"][index].get("name", "")
            if any(node_name == other or node_name.startswith(other + "_") for other in excluded):
                continue
            if node_name == key or node_name.startswith((key + "_", adapter)):
                roots.append(index)
        if not roots:
            raise ValueError(f"No geometry found for {key} in {path.name}")
        overrides = {}
        reverse = rear_reversed and key == "D02_Rear"
        if key in ("D02_Front", "D02_Rear"):
            for component, time in (("backrest", 0), ("canopy", 0), ("legrest", 0 if reverse else 1)):
                overrides.update(seat_keyframe(source, source_binary, f"{key}_{component}", time))
        flat = []

        def flatten(index, parent):
            original = source["nodes"][index]
            node = {**original, **overrides.get(index, {})}
            matrix = parent @ node_matrix(node)
            if reverse and original.get("name") == key:
                matrix = matrix @ np.diag((-1., 1., -1., 1.))
            if "mesh" in node:
                mesh = source["meshes"][node["mesh"]]
                weights = node.pop("weights", mesh.get("weights", ()))
                if "weights" in mesh or any("targets" in p for p in mesh["primitives"]):
                    node["mesh"] = apply_morph(source, source_binary, node["mesh"], weights)
                node.pop("children", None)
                set_matrix(node, matrix)
                flat.append(node)
            for child in original.get("children", ()):
                flatten(child, matrix)

        for root in roots:
            flatten(root, np.eye(4))
        if not flat:
            raise ValueError(f"No renderable meshes for {key}")
        doc, binary = compact_active_scene(
            {**source, "nodes": flat, "scenes": [{"nodes": list(range(len(flat)))}], "scene": 0},
            source_binary,
        )
        parts = scene_parts(doc, binary, station=stations[0])
        result.append(CabinModule(key, name, cabin_id, stations, doc, binary, parts))
    return result


def station_matrix(datum):
    cos, sin = math.cos(datum.angle), math.sin(datum.angle)
    x, y, z = datum.center
    return np.array([[cos, -sin, 0, x], [sin, cos, 0, y],
                     [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


def delta_transform(open_datums, datums, station):
    return station_matrix(datums[station]) @ np.linalg.inv(station_matrix(open_datums[station]))


def move_point(matrix, point):
    return matrix[:3, :3] @ point + matrix[:3, 3]


@lru_cache(maxsize=3)
def frame_sweep(width, wheel, ports, steps):
    samples = prepare_fold_samples(width, wheel, ports, steps=steps)
    return [(phase, doc, binary, datums, scene_parts(doc, binary)) for phase, doc, binary, datums in samples]


def scan_fold(modules, sweep):
    reference = sweep[0][3]
    posed = []
    for phase, _, _, datums, frame in sweep:
        occupied = {}
        blocked = []
        for module in modules:
            base, *extra_stations = module.stations
            delta = delta_transform(reference, datums, base)
            occupied[module.key] = pose_parts(module.parts, delta)
            if phase == 3 and (hit := ground_collision(module.key, occupied[module.key], phase)):
                blocked.append(hit)
            for side in extra_stations:
                moved = move_point(delta, reference[side].center)
                if np.linalg.norm(moved - datums[side].center) > .009:
                    blocked.append(Collision(module.key, None, "多点接口", "另一固定接口",
                                             tuple(float(x) for x in datums[side].center), phase))
        posed.append(FoldSample(phase, frame, occupied, tuple(blocked)))
    return choose_fold_order(posed), {sample.stage: sample for sample in posed}


def append_module(doc, binary, module, transform):
    part = deepcopy(module.doc)
    offsets = {key: len(doc[key]) for key in ("bufferViews", "accessors", "materials", "meshes", "nodes")}
    while len(binary) % 4:
        binary.append(0)
    start = len(binary)
    binary.extend(module.binary)
    for view in part["bufferViews"]:
        view["byteOffset"] = start + view.get("byteOffset", 0)
    for field in part["accessors"]:
        field["bufferView"] += offsets["bufferViews"]
    for mesh in part["meshes"]:
        for primitive in mesh["primitives"]:
            primitive["attributes"] = {key: value + offsets["accessors"] for key, value in primitive["attributes"].items()}
            if "indices" in primitive:
                primitive["indices"] += offsets["accessors"]
            if "material" in primitive:
                primitive["material"] += offsets["materials"]
    for node in part["nodes"]:
        node["mesh"] += offsets["meshes"]
        set_matrix(node, transform @ node_matrix(node))
    for key in offsets:
        doc[key].extend(part[key])
    roots = doc["scenes"][doc.get("scene", 0)]["nodes"]
    roots.extend(range(offsets["nodes"], len(doc["nodes"])))


def bounds(parts):
    return np.min([part.bounds[0] for part in parts], axis=0), np.max([part.bounds[1] for part in parts], axis=0)


def parked_transform(module, frame_bounds, put_on_left):
    low, high = bounds(module.parts)
    frame_low, frame_high = frame_bounds
    # Leave the mounts visible and keep the removed modules off the frame.
    x = frame_low[0] - .20 - high[0] if put_on_left else frame_high[0] + .20 - low[0]
    matrix = np.eye(4)
    matrix[:3, 3] = (x, -low[1], .025 - (low[2] + high[2]) / 2)
    return matrix


def component_name(node, material, cabin_id=""):
    if cabin_id == "S01" and material in ("D02_upholstery", "D02_canopy", "D02_canopy_stay"):
        return "布兜"
    source = (node + " " + material).lower()
    for text, words in (
        ("腿托", ("legrest", "kneecover")), ("头篷", ("canopy", "pet_mesh")),
        ("靠背", ("backrest",)), ("护栏", ("guard",)),
        ("侧接口", ("mount", "receiver", "dock", "host", "adapter")),
        ("推把", ("handle", "grip")), ("后横杆", ("rear_cross", "rear_bar", "rearbar", "park_brake")),
        ("前横杆", ("front_cross", "front_bar", "frontbar")), ("车轮", ("tire", "spoke", "sealed_hub")),
        ("前轮叉", ("fork", "caster")), ("后支撑梁", ("rear_leg", "rear_arm")),
        ("主梁", ("main_lower", "main_upper", "main_beam")), ("折叠关节", ("pivot", "hinge")),
        ("脚踏", ("footrest",)), ("座舱外壳", ("shell", "upholstery", "fabric", "liner")),
        ("落地支脚", ("ground_pad", "standing", "stand_foot")),
        ("地面", ("ground",)),
    ):
        if any(word in source for word in words):
            wheel_corner = {"FL": "左前", "FR": "右前", "RL": "左后", "RR": "右后"}.get(node.split("_")[0])
            if text == "车轮" and wheel_corner:
                return wheel_corner + "轮"
            return text
    return "车架部件" if source.strip() else "固定点"


def removal_witness(module, decision):
    contacts = decision.contacts
    return next((hit for hit in contacts if hit.group == module.key and not hit.other_group),
                next((hit for hit in contacts
                      if module.key in (hit.group, hit.other_group)
                      and any(key in decision.retained for key in (hit.group, hit.other_group))), None))


def removal_summary(module, hit, modules):
    stage = ("开始收折时" if hit.stage < 1.15 else
             "折叠途中" if hit.stage < 2.8 else "接近完全折叠时")
    if hit.source == "多点接口":
        return {"name": module.name, "reason": "收折时，两处固定接口的相对位置会改变。", "zone": "固定接口"}
    if hit.target == "ground":
        part = component_name(hit.source, hit.source_material, module.cabin_id)
        return {"name": module.name, "reason": f"折叠后站立时，{part}会先碰到地面。",
                "zone": f"{part}与地面"}
    source, target = ((hit.source, hit.source_material), (hit.target, hit.target_material))
    peer = hit.other_group
    if hit.other_group == module.key:
        source, target = target, source
        peer = hit.group
    source_name = component_name(*source, cabin_id=module.cabin_id)
    target_name = component_name(*target, cabin_id=modules[peer].cabin_id if peer else "")
    destination = f"{modules[peer].name.split(' · ')[0]}的{target_name}" if peer else target_name
    return {"name": module.name, "reason": f"{stage}，{source_name}会碰到{destination}。",
            "zone": f"{source_name}与{destination}"}


def marker_location(module, hit, poses, display):
    source = poses[hit.stage]
    if hit.source == "多点接口":
        return None
    if hit.target == "ground":
        return np.asarray(hit.point) + (0, .025, 0) if display.stage == 3 else None
    node, material = hit.target, hit.target_material
    peer = hit.other_group
    if peer == module.key:
        node, material, peer = hit.source, hit.source_material, hit.group
    point = np.asarray(hit.point)
    original = source.cabins[peer] if peer else source.frame
    displayed = display.cabins[peer] if peer else display.frame

    def match(part):
        return part.name.split(".")[0] == node.split(".")[0] and part.material == material

    old = next((part.triangles.reshape(-1, 3) for part in original if match(part)), None)
    new = next((part.triangles.reshape(-1, 3) for part in displayed if match(part)), None)
    if old is None or new is None:
        return None
    if old.shape != new.shape:
        return new.mean(axis=0)
    return new[np.argmin(np.sum((old - point) ** 2, axis=1))]


def add_marker(doc, binary, name, point):
    marker = Geometry()
    for x, z in ((0, 1), (0, 2), (1, 2)):
        ring = []
        for i in range(25):
            angle = i * math.tau / 24
            position = np.array(point)
            position[x] += .032 * math.cos(angle)
            position[z] += .032 * math.sin(angle)
            ring.append(tuple(position))
        reference = tuple(float(axis not in (x, z)) for axis in range(3))
        marker.tube(ring, .003, reference, sides=6, capped=False)
    result = mesh_node(doc, binary, name, {"fold_contact": marker})
    doc["scenes"][doc.get("scene", 0)]["nodes"].append(result)


def export_poses(output, mode, spec, names, width, wheel, *, steps):
    identity = spec["id"]
    ports = tuple(port[0] for port in spec["ports"])
    folder = output / f"{mode}_cabin"
    assembled = ROOT / "models" / f"{mode}_cabin" / f"{identity}_w{width}_state1_wheel_{wheel}.glb"
    layout = module_layout(mode, spec, names)
    modules = cabin_modules(assembled, layout)
    sweep = frame_sweep(width, wheel, ports, steps)
    decision, poses = scan_fold(modules, sweep)
    note = ""
    if identity == "D02":
        reversed_modules = cabin_modules(assembled, layout, rear_reversed=True)
        reversed_decision, reversed_poses = scan_fold(reversed_modules, sweep)
        if len(reversed_decision.retained) > len(decision.retained):
            rear = modules[1]
            hit = removal_witness(rear, decision)
            same_direction = removal_summary(rear, hit, {module.key: module for module in modules})["reason"]
            note = ("两舱同向时，上舱也要拆。" + same_direction +
                    "要保留上舱，先拆下舱，再把上舱反装，靠背调至 95°，头篷和腿托调到第 1 档。")
            modules, decision, poses = reversed_modules, reversed_decision, reversed_poses
        else:
            note = "按两舱同向、靠背 95°、头篷第 1 档、腿托收起检查。"
            rear = reversed_modules[1]
            hit = removal_witness(rear, reversed_decision)
            if hit is not None and rear.key in reversed_decision.removed:
                reason = removal_summary(rear, hit, {module.key: module for module in reversed_modules})["reason"]
                note += "上舱反装也需要拆下。" + reason
    elif "S15" in (identity, spec.get("lower"), spec.get("upper")):
        note = "软载货舱按空舱检查，收折前需清空。"
    by_key = {module.key: module for module in modules}
    reference = sweep[0][3]
    removed = [(by_key[key], removal_witness(by_key[key], decision)) for key in decision.removed]
    if any(hit is None for _, hit in removed):
        raise ValueError(f"Missing reason for cabin removal in {assembled.name}")
    summary = {
        "removed": [removal_summary(module, hit, by_key) for module, hit in removed],
        "retained": [{"name": by_key[key].name} for key in decision.retained],
        "sampleCount": decision.checked_samples,
    }
    if note:
        summary["note"] = note
    folder.mkdir(parents=True, exist_ok=True)
    for phase, frame, frame_bin, datums, geometry in sweep:
        if phase not in STAGES:
            continue
        doc, binary = deepcopy(frame), bytearray(frame_bin)
        scene = doc["scenes"][doc.get("scene", 0)]
        frame_bounds = bounds(geometry + [part for key in decision.retained for part in poses[phase].cabins[key]])
        for module in modules:
            if module.key in decision.retained:
                delta = delta_transform(reference, datums, module.stations[0])
                append_module(doc, binary, module, delta)
        markers = []
        detached = []
        for i, (module, hit) in enumerate(removed):
            parked = parked_transform(module, frame_bounds, i > 0)
            append_module(doc, binary, module, parked)
            low, high = bounds(module.parts)
            label = move_point(parked, ((low[0] + high[0]) / 2, high[1] + .043, (low[2] + high[2]) / 2))
            detached.append({**summary["removed"][i], "position": label.tolist()})
            point = marker_location(module, hit, poses, poses[phase])
            if point is not None and all(np.linalg.norm(point - other) > .038 for other in markers):
                add_marker(doc, binary, f"FoldContact_{i + 1}", point)
                markers.append(point)
        scene.setdefault("extras", {})["strollerFold"] = {**summary, "removed": detached, "hasMarker": bool(markers)}
        doc["buffers"][0]["byteLength"] = len(binary)
        write_glb(folder / f"{identity}_w{width}_state{int(phase)}_wheel_{wheel}.glb", doc, binary)
    status = ", ".join(f"{module.key}: {round(hit.stage, 3)} {hit.source}/{hit.target}" for module, hit in removed)
    print(f"{identity} {width} {wheel}: keep {len(decision.retained)}/{len(modules)} {status}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--cabin", action="append")
    parser.add_argument("--width", action="append", type=int, choices=(380, 550, 650))
    parser.add_argument("--wheel", action="append", choices=tuple(SPECS))
    parser.add_argument("--steps", type=int, default=16, help="sweep intervals per authored fold segment")
    args = parser.parse_args()
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
        export_poses(args.output, mode, spec, names, width, wheel, steps=args.steps)


if __name__ == "__main__":
    main()
