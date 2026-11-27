"""Prepare the leveled display chassis and its standard receiver datums."""

from collections import Counter
import math

from create_d02_double_seat import (
    D02_RISE, FRONT_PORT_ADVANCE, FRONT_PORT_DROP, HIGH_PORT_RISE, HIGH_PORT_SETBACK,
)
from d02_front_mount import relocate_front_ports, relocate_high_ports
from generate_wheel_variants import (
    DEFAULT_WHEEL, MODEL_DIR, apply_wheel_dimensions, compact_active_scene, read_glb, transform_vec3,
)
from prepare_d02_chassis import fit_wheel_clearance, level_assembly, prepare_chassis, replace_positions
from universal_mount import remove_receiver_nodes, standardize_receivers


def _bake_narrow_frame(doc, binary):
    """Bake the narrow CAD export's per-part transforms before fitting wheels."""
    position_normals = {"POSITION", "NORMAL"}
    mesh_uses = Counter(node["mesh"] for node in doc["nodes"] if "mesh" in node)
    attribute_uses = Counter(
        accessor for mesh in doc["meshes"] for primitive in mesh["primitives"]
        for semantic, accessor in primitive["attributes"].items()
        if semantic in position_normals
    )
    for node in doc["nodes"]:
        if "matrix" in node:
            raise ValueError("Narrow chassis export must use per-part translation, rotation and scale")
        if not {"rotation", "translation", "scale"}.intersection(node):
            continue
        mesh = node["mesh"]
        if mesh_uses[mesh] != 1:
            raise ValueError(f"Narrow chassis part shares its transformed mesh: {node['name']}")
        x, y, z, w = node.pop("rotation", (0, 0, 0, 1))
        axes = (
            (1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
            (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
            (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)),
        )
        tx, ty, tz = node.pop("translation", (0, 0, 0))
        sx, sy, sz = node.pop("scale", (1, 1, 1))

        def rotate(a, b, c):
            return tuple(u*a + v*b + q*c for u, v, q in axes)

        def position(p):
            a, b, c = rotate(p[0]*sx, p[1]*sy, p[2]*sz)
            return a+tx, b+ty, c+tz

        def normal(n):
            rotated = rotate(n[0]/sx, n[1]/sy, n[2]/sz)
            length = math.hypot(*rotated) or 1
            return tuple(v / length for v in rotated)

        for primitive in doc["meshes"][mesh]["primitives"]:
            attributes = primitive["attributes"]
            for semantic in position_normals.intersection(attributes):
                if attribute_uses[attributes[semantic]] != 1:
                    raise ValueError(f"Narrow chassis part shares transformed vertices: {node['name']}")
            replace_positions(doc, binary, attributes["POSITION"], position)
            if "NORMAL" in attributes:
                transform_vec3(doc, binary, attributes["NORMAL"], normal)


def _prepare_narrow_frame(wheel):
    default_wheel = DEFAULT_WHEEL[380]
    source, source_binary = read_glb(MODEL_DIR / f"w380_state1_wheel_{default_wheel}.glb")
    remove_receiver_nodes(source)
    doc, binary = compact_active_scene(source, source_binary)
    _bake_narrow_frame(doc, binary)
    parts = {node["name"].split(".")[0]: i for i, node in enumerate(doc["nodes"])}
    rotating = {
        name: i for name, i in parts.items()
        if name.split("_", 1)[-1].startswith(("tire", "sealed_hub", "spoke_"))
    }
    apply_wheel_dimensions(doc, binary, list(rotating.values()), default_wheel, wheel)
    fit_wheel_clearance(doc, binary, {name: doc["nodes"][i] for name, i in parts.items()}, rotating)
    return doc, binary


def prepare_catalog_frame(width, wheel, ports=()):
    """Fit the selected wheelset and retain receiver coordinates before leveling."""
    doc, binary = _prepare_narrow_frame(wheel) if width == 380 else prepare_chassis(width, wheel)
    if width != 380 and tuple(ports) == ("L", "H"):
        relocate_front_ports(doc, binary, forward=FRONT_PORT_ADVANCE, down=FRONT_PORT_DROP)
        relocate_high_ports(doc, binary, aft=HIGH_PORT_SETBACK, up=HIGH_PORT_RISE)
    centers = standardize_receivers(doc, binary, rise=D02_RISE)
    level = level_assembly(doc, binary)
    return doc, binary, centers, level
