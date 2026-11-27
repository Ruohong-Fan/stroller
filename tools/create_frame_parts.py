"""Export standalone frame components from the assembled chassis CAD scenes."""

from pathlib import Path

from catalog_frame import _bake_narrow_frame
from generate_wheel_variants import (
    DEFAULT_WHEEL, MODEL_DIR, SPECS, compact_active_scene, read_glb, write_glb,
)
from prepare_d02_chassis import replace_positions


FRAME_WIDTH_NAMES = {380: "窄档", 550: "中档", 650: "宽档"}
WHEEL_NAMES = {"S": "小轮", "M": "中轮", "L": "大轮"}

FRONT_WHEEL_MOUNTS = tuple(
    f"front_light_elastomer_{{side}}_{i}" for i in range(3)
) + (
    "vertical_front_guide_{side}", "vertical_front_spigot_{side}",
)
REAR_WHEEL_MOUNTS = tuple(
    f"{stem}_{{side}}" for stem in (
        "aligned_short_rear_arm", "compact_rear_elastomer_case",
        "fixed_rear_module_root", "flush_rear_case_cap",
        "rear_brake_drive_input", "rear_load_spigot", "rear_module_shoulder",
        "V46_internal_lock_actuation", "V46_internal_module_brake_link",
    )
)
BRAKE_PEDAL = (
    "V46_brake_lever", "V46_brake_pivot", "V46_brake_tread",
    "V46_tread_rib_0", "V46_tread_rib_1", "V46_tread_rib_-1",
)

# The right-side CAD datum suffix is +1 and the left-side suffix is -1.
WHEELS = (
    ("front-left", "左前轮", "FL", -1, FRONT_WHEEL_MOUNTS, ()),
    ("front-right", "右前轮", "FR", 1, FRONT_WHEEL_MOUNTS, ()),
    ("rear-left", "左后轮", "RL", -1, REAR_WHEEL_MOUNTS, BRAKE_PEDAL),
    ("rear-right", "右后轮", "RR", 1, REAR_WHEEL_MOUNTS, ()),
)
CROSSBARS = (
    ("front-crossbar", "前横杆", "连接左右前轮安装座。", ("front_fixed_round_tube",)),
    ("rear-crossbar", "后横杆", "可拆卸的后横杆外管。", ("V46_rearbar",)),
    ("brake-shaft", "后横杆制动轴", "位于后横杆内部，传递停车制动动作。", ("enclosed_brake_shaft",)),
    ("push-grip", "推把横杆", "握持横杆及两端的收车支撑垫。", ("round_handle", "ground_pad_1", "ground_pad_-1")),
)
BEAMS = (
    ("main-beam-lower", "主梁下段", "从前轮安装区通向折叠关节。两侧采用相同梁管。",
     ("main_lower_1",)),
    ("main-beam-upper", "主梁上段", "从折叠关节通向推把。两侧采用相同梁管。",
     ("main_upper_1",)),
    ("rear-support-beam", "后支撑副梁", "连接折叠关节与后轮安装区。两侧采用相同梁管。",
     ("V46_rear_leg_1",)),
    ("handle-slider", "伸缩推把侧梁", "沿主梁上段滑动，调节推把高度。两侧采用相同梁管。",
     ("handle_slider_1",)),
)


def _read_chassis(source, width):
    doc, binary = read_glb(source / f"w{width}_state1_wheel_{DEFAULT_WHEEL[width]}.glb")
    if width == 380:
        # The narrow export stores each member in local coordinates; the 550
        # and 650 exports already have their transforms baked.
        _bake_narrow_frame(doc, binary)
    scene = doc["scenes"][doc.get("scene", 0)]
    indices = {doc["nodes"][i]["name"].split(".")[0]: i for i in scene["nodes"]}
    return doc, binary, indices


def _write_part(output, frame, category, part_id, name, desc, members):
    source_doc, source_binary, indices = frame
    selection = {**source_doc, "scenes": [{"nodes": [indices[member] for member in members]}], "scene": 0}
    doc, binary = compact_active_scene(selection, source_binary)

    positions = {
        primitive["attributes"]["POSITION"]
        for mesh in doc["meshes"]
        for primitive in mesh["primitives"]
    }
    accessors = [doc["accessors"][i] for i in positions]
    center = tuple(
        (min(accessor["min"][axis] for accessor in accessors)
         + max(accessor["max"][axis] for accessor in accessors)) / 2
        for axis in range(3)
    )
    for accessor in positions:
        replace_positions(doc, binary, accessor, lambda p: tuple(v - c for v, c in zip(p, center)))
    doc["scenes"][0]["name"] = name
    doc["scenes"][0]["extras"] = {"partId": part_id, "category": category}
    path = f"other/{category}/{part_id}"
    glb = output / (path + ".glb")
    glb.parent.mkdir(parents=True, exist_ok=True)
    write_glb(glb, doc, binary)
    return {"id": part_id, "name": name, "path": path, "description": desc}


def build_frame_parts(output: Path, *, source=MODEL_DIR):
    frames = {width: _read_chassis(source, width) for width in DEFAULT_WHEEL}
    catalogue = {"wheels": [], "crossbars": [], "beams": []}
    for width, wheel in DEFAULT_WHEEL.items():
        frame = frames[width]
        _, _, indices = frame
        for position, position_name, prefix, side, mounts, extras in WHEELS:
            front = position.startswith("front-")
            diameter = SPECS[wheel]["front_dia" if front else "rear_dia"]
            tread = SPECS[wheel]["front_width" if front else "rear_width"]
            feature = "含转向轮叉、上置弹性件和快拆插接件。" if front else "含减震摆臂和停车制动接口。"
            if extras:
                feature = "含减震摆臂和停车制动踏板。"
            rotating_and_axle = [member for member in indices if member.startswith(prefix + "_")]
            if not rotating_and_axle:
                raise ValueError(f"Missing {prefix} wheel on the {width} mm chassis")
            members = [*rotating_and_axle, *(m.format(side=side) for m in mounts), *extras]
            catalogue["wheels"].append(_write_part(
                output, frame, "wheels", f"wheel-{position}-{wheel.lower()}",
                f"{WHEEL_NAMES[wheel]} · {position_name}",
                f"直径 {diameter} mm，胎宽 {tread} mm。{feature}", members,
            ))

    for part, name, desc, members in CROSSBARS:
        for width, width_name in FRAME_WIDTH_NAMES.items():
            catalogue["crossbars"].append(_write_part(
                output, frames[width], "crossbars", f"{part}-{width}",
                f"{name} · {width_name} {width} mm", desc, members,
            ))

    for part, name, desc, members in BEAMS:
        catalogue["beams"].append(_write_part(
            output, frames[550], "beams", part, name, desc, members,
        ))
    return catalogue
