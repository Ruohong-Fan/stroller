"""Fit and level the D02 display chassis for the selected wheelset."""

from __future__ import annotations

import math

from generate_wheel_variants import (
    DEFAULT_WHEEL,
    MODEL_DIR,
    apply_wheel_dimensions,
    compact_active_scene,
    position_accessor_for_node,
    read_glb,
    read_positions,
    transform_vec3,
)


TREAD_CLEARANCE = 0.004


def bounds(doc, node):
    positions = doc["accessors"][position_accessor_for_node(doc, node)]
    return positions["min"], positions["max"]


def replace_positions(doc, binary, index, transform):
    values = transform_vec3(doc, binary, index, transform)
    doc["accessors"][index]["min"] = [min(axis) for axis in zip(*values)]
    doc["accessors"][index]["max"] = [max(axis) for axis in zip(*values)]


def fit_along_axis(doc, binary, node, axis, scale, offset):
    def position(vector):
        value = list(vector)
        value[axis] = vector[axis] * scale + offset
        return value

    def normal(vector):
        value = list(vector)
        value[axis] /= scale
        length = math.sqrt(sum(component * component for component in value)) or 1.0
        return [component / length for component in value]

    for primitive in doc["meshes"][node["mesh"]]["primitives"]:
        attributes = primitive["attributes"]
        replace_positions(doc, binary, attributes["POSITION"], position)
        if scale != 1 and "NORMAL" in attributes:
            transform_vec3(doc, binary, attributes["NORMAL"], normal)


def fit_wheel_clearance(doc, binary, nodes, rotating_nodes):
    """Keep the suspension mounts fixed and clear the tire running surfaces."""
    for wheel, lateral_sign in (("FL", 1), ("FR", -1)):
        tire_min, tire_max = bounds(doc, nodes[f"{wheel}_tire"])
        bridge = nodes[f"{wheel}_fork_bridge"]
        bridge_min, bridge_max = bounds(doc, bridge)
        fork = nodes[f"{wheel}_fork"]
        fork_min, fork_max = bounds(doc, fork)
        drop = tire_max[1] - bridge_min[1] + TREAD_CLEARANCE
        if drop > 0:
            # Extend the fork towards the axle, keeping its top at the bridge.
            scale = 1 + drop / (fork_max[1] - fork_min[1])
            fit_along_axis(doc, binary, fork, 1, scale, fork_max[1] * (1 - scale))
            fit_along_axis(doc, binary, nodes[f"{wheel}_captive_axle"], 1, 1, -drop)
            for name in rotating_nodes:
                if name.startswith(wheel + "_"):
                    fit_along_axis(doc, binary, nodes[name], 1, 1, -drop)

        # Give the wider tread space by moving the fork towards the frame
        # centerline and extending the inboard side of the steering bridge.
        inner_tread = min(lateral_sign * tire_min[2], lateral_sign * tire_max[2])
        outer_fork = max(lateral_sign * fork_min[2], lateral_sign * fork_max[2])
        shift = max(0, outer_fork - inner_tread + TREAD_CLEARANCE)
        if shift > 0:
            fit_along_axis(doc, binary, fork, 2, 1, -lateral_sign * shift)
        outer_bridge = bridge_max[2] if lateral_sign > 0 else bridge_min[2]
        scale = 1 + (shift + 0.006) / (bridge_max[2] - bridge_min[2])
        fit_along_axis(doc, binary, bridge, 2, scale, outer_bridge * (1 - scale))

    for wheel, lateral_sign in (("RL", 1), ("RR", -1)):
        side = -lateral_sign
        fixed_parts = (
            f"fixed_rear_module_root_{side}", f"rear_module_shoulder_{side}",
            f"rear_load_spigot_{side}", f"rear_brake_drive_input_{side}",
            f"compact_rear_elastomer_case_{side}", f"aligned_short_rear_arm_{side}",
            f"{wheel}_brake_backplate",
        )
        tire_min, tire_max = bounds(doc, nodes[f"{wheel}_tire"])
        inner_tread = min(lateral_sign * tire_min[2], lateral_sign * tire_max[2])
        outer_mount = max(lateral_sign * v[2] for name in fixed_parts for v in bounds(doc, nodes[name]))
        offset = outer_mount - inner_tread + TREAD_CLEARANCE
        if offset <= 0:
            continue
        # Keep the brake and inboard axle end seated in the rear arm while
        # extending the axle and wheel hub away from the module body.
        axle = nodes[f"{wheel}_fixed_axle"]
        axle_min, axle_max = bounds(doc, axle)
        inner_end = axle_min[2] if lateral_sign > 0 else axle_max[2]
        scale = 1 + offset / (axle_max[2] - axle_min[2])
        fit_along_axis(doc, binary, axle, 2, scale, inner_end * (1 - scale))
        for name in rotating_nodes:
            if name.startswith(wheel + "_"):
                fit_along_axis(doc, binary, nodes[name], 2, 1, lateral_sign * offset)


def prepare_chassis(width: int, wheel: str) -> tuple[dict, bytearray]:
    original_wheel = DEFAULT_WHEEL[width]
    path = MODEL_DIR / f"w{width}_state1_wheel_{original_wheel}.glb"
    doc, binary = compact_active_scene(*read_glb(path))
    # Resize the rotating pieces. The original fixed axle, bearings and
    # suspension pieces must stay in contact with their frame mounts.
    nodes = {node["name"].split(".")[0]: node for node in doc["nodes"]}
    rotating_nodes = {
        node["name"].split(".")[0]: index for index, node in enumerate(doc["nodes"])
        if node["name"].split(".")[0].split("_", 1)[-1].startswith(("tire", "sealed_hub", "spoke_"))
    }
    apply_wheel_dimensions(doc, binary, list(rotating_nodes.values()), original_wheel, wheel)
    fit_wheel_clearance(doc, binary, nodes, rotating_nodes)
    return doc, binary


def level_assembly(doc: dict, binary: bytearray) -> tuple[float, float]:
    """Put the front and rear tread planes at zero while keeping all joints fixed."""
    def tire_positions(prefix):
        node = next(node for node in doc["nodes"] if node["name"].startswith(prefix))
        return read_positions(doc, binary, position_accessor_for_node(doc, node))

    front_tread = tire_positions("FL_tire")
    rear_tread = tire_positions("RL_tire")

    def lowest_point(vertices, angle):
        sine, cosine = math.sin(angle), math.cos(angle)
        return min(x * sine + y * cosine for x, y, _ in vertices)

    low, high = -0.1, 0.1
    for _ in range(24):
        angle = (low + high) / 2
        if lowest_point(rear_tread, angle) < lowest_point(front_tread, angle):
            low = angle
        else:
            high = angle
    angle = (low + high) / 2
    height = min(lowest_point(front_tread, angle), lowest_point(rear_tread, angle))
    if abs(angle) < 1e-6 and abs(height) < 1e-6:
        return 0.0, 0.0

    sine, cosine = math.sin(angle), math.cos(angle)
    positions, normals = set(), set()
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            positions.add(attributes["POSITION"])
            if "NORMAL" in attributes:
                normals.add(attributes["NORMAL"])

    # Bake the transform so the viewer's bounds reflect the exact tread plane.
    for index in positions:
        replace_positions(doc, binary, index, lambda p: (
            p[0] * cosine - p[1] * sine,
            p[0] * sine + p[1] * cosine - height,
            p[2],
        ))
    for index in normals:
        transform_vec3(doc, binary, index, lambda n: (
            n[0] * cosine - n[1] * sine,
            n[0] * sine + n[1] * cosine,
            n[2],
        ))
    return angle, height
