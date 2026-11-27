"""Check sampled folding poses against the triangles in the published models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import combinations
import re

import numpy as np


CONTACT_EPSILON = 1e-7  # metres; compensate for the glTF FLOAT coordinates
ACCESSOR_TYPES = {5121: "<u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
ACCESSOR_SIZES = {"SCALAR": 1, "VEC3": 3}
RECEIVER_NAMES = re.compile(
    r"^(?:Universal_Cabin_Receiver_([HML])_|narrow-open_host_-?1_([HML])(?:\.|$))"
)
ADAPTER_NAMES = re.compile(r"^D02_WideAdapter_([HML])_")
MOUNT_MATERIALS = frozenset(
    ("D02_mount_shoe", "D02_mount_pin", "D02_mounts", "D02_mount_cheek", "D02_width_adapter")
)


@dataclass(frozen=True)
class MeshPart:
    path: tuple[str, ...]
    material: str
    triangles: np.ndarray
    station: str | None = None

    @property
    def name(self):
        return self.path[-1]

    @cached_property
    def face_bounds(self):
        return self.triangles.min(axis=1), self.triangles.max(axis=1)

    @cached_property
    def bounds(self):
        low, high = self.face_bounds
        return low.min(axis=0), high.max(axis=0)


@dataclass(frozen=True)
class Collision:
    group: str
    other_group: str | None
    source: str
    target: str
    point: tuple[float, float, float]
    stage: float
    source_material: str = ""
    target_material: str = ""

    def as_dict(self):
        return {
            "group": self.group, "otherGroup": self.other_group,
            "source": self.source, "target": self.target,
            "sourceMaterial": self.source_material, "targetMaterial": self.target_material,
            "point": self.point, "stage": self.stage,
        }


@dataclass(frozen=True)
class FoldSample:
    stage: float
    frame: list[MeshPart]
    cabins: dict[str, list[MeshPart]]
    blocked: tuple[Collision, ...] = ()


@dataclass(frozen=True)
class FoldDecision:
    retained: tuple[str, ...]
    removed: tuple[str, ...]
    contacts: tuple[Collision, ...]
    checked_samples: int


def _access(doc, binary, index):
    field = doc["accessors"][index]
    view = doc["bufferViews"][field["bufferView"]]
    dtype = np.dtype(ACCESSOR_TYPES[field["componentType"]])
    size = ACCESSOR_SIZES[field["type"]]
    return np.ndarray(
        (field["count"], size), dtype, buffer=binary,
        offset=view.get("byteOffset", 0) + field.get("byteOffset", 0),
        strides=(view.get("byteStride", dtype.itemsize * size), dtype.itemsize),
    )


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T

    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, :3] *= node.get("scale", (1.0, 1.0, 1.0))
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def _station(node, fallback):
    name = node.get("name", "")
    match = RECEIVER_NAMES.match(name) or ADAPTER_NAMES.match(name)
    if match:
        return next(group for group in match.groups() if group)
    return node.get("extras", {}).get("station", fallback)


def scene_parts(doc, binary, roots=None, *, station=None):
    """Read indexed or unindexed glTF triangles, applying node TRS and default morph weights."""
    scene = doc["scenes"][doc.get("scene", 0)]
    result = []

    def visit(index, matrix, path, site):
        node = doc["nodes"][index]
        matrix = matrix @ node_matrix(node)
        path = (*path, node.get("name", f"node-{index}"))
        site = _station(node, site)
        if "mesh" in node:
            mesh = doc["meshes"][node["mesh"]]
            weights = node.get("weights", mesh.get("weights", ()))
            for p in mesh["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                points = _access(doc, binary, p["attributes"]["POSITION"]).astype(float)
                for weight, target in zip(weights, p.get("targets", ())):
                    if weight and "POSITION" in target:
                        points += weight * _access(doc, binary, target["POSITION"])
                points = points @ matrix[:3, :3].T + matrix[:3, 3]
                if "indices" in p:
                    points = points[_access(doc, binary, p["indices"]).ravel()]
                material = doc["materials"][p["material"]].get("name", "") if "material" in p else ""
                if points.size:
                    result.append(MeshPart(path, material, points.reshape(-1, 3, 3), site))
        for child in node.get("children", ()):
            visit(child, matrix, path, site)

    for root in scene["nodes"] if roots is None else roots:
        visit(root, np.eye(4), (), station)
    return result


def pose_parts(parts, matrix):
    """Apply one rigid world-space transform to a group of collision meshes."""
    return [replace(part, triangles=part.triangles @ matrix[:3, :3].T + matrix[:3, 3]) for part in parts]


def _intersect(a, b):
    """Use triangle separating axes on already-filtered candidate pairs."""
    ea = np.roll(a, -1, axis=1) - a
    eb = np.roll(b, -1, axis=1) - b
    na = np.cross(ea[:, 0], ea[:, 1])
    nb = np.cross(eb[:, 0], eb[:, 1])
    axes = [na, nb]
    axes.extend(np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3))
    # These in-plane axes distinguish coplanar fabric and shell triangles.
    axes.extend(np.cross(n, edges[:, i]) for n, edges in ((na, ea), (nb, eb)) for i in range(3))
    hits = (np.linalg.norm(na, axis=1) > 1e-12) & (np.linalg.norm(nb, axis=1) > 1e-12)
    for axis in axes:
        x = np.einsum("nij,nj->ni", a, axis)
        y = np.einsum("nij,nj->ni", b, axis)
        epsilon = np.linalg.norm(axis, axis=1) * CONTACT_EPSILON
        hits &= (x.min(axis=1) <= y.max(axis=1) + epsilon) & (y.min(axis=1) <= x.max(axis=1) + epsilon)
        if not hits.any():
            break
    return np.flatnonzero(hits)


def _triangle_edges(triangle):
    return zip(triangle, np.roll(triangle, -1, axis=0))


def _coplanar_witness(a, b):
    """Find a vertex inside the other face or a crossing of two in-plane edges."""
    normal = np.cross(a[1] - a[0], a[2] - a[0])
    axes = [index for index in range(3) if index != int(np.argmax(np.abs(normal)))]

    def det(u, v):
        return u[axes[0]] * v[axes[1]] - u[axes[1]] * v[axes[0]]

    for first, second in ((a, b), (b, a)):
        for point in first:
            side = [det(end-start, point-start) for start, end in _triangle_edges(second)]
            if min(side) >= 0 or max(side) <= 0:
                return point
    for start, end in _triangle_edges(a):
        first = end - start
        for other, limit in _triangle_edges(b):
            second = limit - other
            scale = det(first, second)
            if abs(scale) <= 1e-14:
                continue
            offset = other - start
            s, t = det(offset, second) / scale, det(offset, first) / scale
            if 0 <= s <= 1 and 0 <= t <= 1:
                return start + s * first
    return None


def _surface_witness(a, b):
    """Locate an intersection on a triangle edge for the preview marker."""
    for moving, fixed in ((a, b), (b, a)):
        ab, ac = fixed[1:] - fixed[0]
        normal = np.cross(ab, ac)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        d00 = ab @ ab
        d01 = ab @ ac
        d11 = ac @ ac
        scale = d00 * d11 - d01 * d01
        for start, stop in _triangle_edges(moving):
            vector = stop - start
            step = vector @ normal
            if abs(step) <= 1e-12:
                continue
            time = ((fixed[0] - start) @ normal) / step
            if not -1e-6 <= time <= 1.000001:
                continue
            point = start + time * vector
            offset = point - fixed[0]
            d20, d21 = ab @ offset, ac @ offset
            v = (d11 * d20 - d01 * d21) / scale
            w = (d00 * d21 - d01 * d20) / scale
            if v >= -1e-6 and w >= -1e-6 and v+w <= 1.000001:
                return tuple(float(value) for value in point)
    witness = _coplanar_witness(a, b)
    if witness is not None:
        return tuple(float(value) for value in witness)
    # At the floating-point tolerance boundary report the overlapping bounds.
    lower = np.maximum(a.min(axis=0), b.min(axis=0))
    upper = np.minimum(a.max(axis=0), b.max(axis=0))
    return tuple(float(value) for value in (upper + lower) / 2)


def _contact(a, b):
    al, ah = a.face_bounds
    bl, bh = b.face_bounds
    eps = CONTACT_EPSILON
    a_min, a_max = a.bounds
    b_min, b_max = b.bounds
    ai = np.flatnonzero(((al <= b_max+eps) & (ah >= b_min-eps)).all(axis=1))
    bi = np.flatnonzero(((bl <= a_max+eps) & (bh >= a_min-eps)).all(axis=1))
    for offset in range(0, len(ai), 96):
        subset = ai[offset:offset+96]
        aa, bb = np.where(((al[subset, None] <= bh[None, bi]+eps) &
                           (ah[subset, None] >= bl[None, bi]-eps)).all(axis=2))
        for start in range(0, len(aa), 2048):
            ca = subset[aa[start:start+2048]]
            cb = bi[bb[start:start+2048]]
            hit = _intersect(a.triangles[ca], b.triangles[cb])
            if hit.size:
                index = hit[0]
                return _surface_witness(a.triangles[ca[index]], b.triangles[cb[index]])
    return None


def _mating_contact(a, b):
    """Ignore only the modeled mounting pieces on their own station receiver."""
    return (a.station is not None and a.station == b.station
            and a.material in MOUNT_MATERIALS and RECEIVER_NAMES.match(b.name) is not None)


def first_contact(source, target, *, skip_mating=False):
    """Return the first two intersecting parts and their contact position."""
    if not target:
        return None
    target_low, target_high = np.array([part.bounds for part in target]).transpose(1, 0, 2)
    epsilon = CONTACT_EPSILON
    for a in source:
        low, high = a.bounds
        indices = np.flatnonzero(((high >= target_low-epsilon) & (low <= target_high+epsilon)).all(axis=1))
        for index in indices:
            b = target[index]
            if skip_mating and _mating_contact(a, b):
                continue
            point = _contact(a, b)
            if point is not None:
                return a, b, point
    return None


def ground_collision(group, parts, stage, ground=0.0, *, tolerance=0.002):
    """Flag a cabin that projects past the folded frame's standing surface."""
    if not parts:
        return None
    part = min(parts, key=lambda item: item.bounds[0][1])
    if part.bounds[0][1] >= ground - tolerance:
        return None
    points = part.triangles.reshape(-1, 3)
    point = points[np.argmin(points[:, 1])]
    return Collision(group, None, part.name, "ground", (float(point[0]), ground, float(point[2])),
                     stage, part.material)


def choose_fold_order(samples):
    """Keep the largest clear subset over the sweep, preferring the first cabin on ties."""
    groups = ()
    individual = {}
    paired = {}
    checked = 0
    for sample in samples:
        if not checked:
            groups = tuple(sample.cabins)
        if tuple(sample.cabins) != groups:
            raise ValueError("All folding samples must use the same ordered cabin groups")
        checked += 1
        for hit in sample.blocked:
            if hit.target != "ground":
                individual.setdefault(hit.group, hit)
        for group, parts in sample.cabins.items():
            if group not in individual:
                contact = first_contact(parts, sample.frame, skip_mating=True)
                if contact:
                    a, b, point = contact
                    individual[group] = Collision(group, None, a.name, b.name, point, sample.stage,
                                                  a.material, b.material)
        for hit in sample.blocked:
            individual.setdefault(hit.group, hit)
        for a_group, b_group in combinations(groups, 2):
            key = (a_group, b_group)
            if key not in paired:
                contact = first_contact(sample.cabins[a_group], sample.cabins[b_group])
                if contact:
                    a, b, point = contact
                    paired[key] = Collision(a_group, b_group, a.name, b.name, point, sample.stage,
                                            a.material, b.material)

    if not checked:
        raise ValueError("A folding decision requires sampled poses")
    available = tuple(group for group in groups if group not in individual)
    for size in range(len(available), -1, -1):
        retained = next((choice for choice in combinations(available, size)
                         if all(pair not in paired for pair in combinations(choice, 2))), None)
        if retained is not None:
            break
    contacts = tuple(sorted((*individual.values(), *paired.values()), key=lambda hit: hit.stage))
    return FoldDecision(retained, tuple(group for group in groups if group not in retained), contacts, checked)
