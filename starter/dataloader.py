"""Standalone relational dataloader for the 01Health treatment-planning task.

Each sample is one case paired with one instruction. The loader reads the task
files directly, builds the original point/geometry/instruction tensors, then
adds relational tooth, arch, same-arch, opposite-arch, and contralateral
features used by the final multimodal model.

Important: features describe relationships in the current scan. They are not a
substitute for a clinically defined desired final arch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import numpy as np

from contract import (
    IDENTITY_Q, case_instructions, geodesic_deg, jaw_of, load_case, load_gold,
    qconj, qmul, qnormalize, yaw_quat,
)
from parser import parse_instruction_keywords

EPS = 1e-8
PARSER_FEATURE_DIM = 17
GOLD_ZERO_TOL_MM = 1e-6
GOLD_ZERO_TOL_DEG = 1e-6


def _load_llm_plan(case_dir: Path, instruction_id: str) -> tuple[set[int], set[int]] | None:
    """Read movement_plan.json and return (move_teeth, fixed_teeth) for an instruction."""
    plan_path = Path(case_dir) / "movement_plan.json"
    if not plan_path.exists():
        return None
    try:
        entries = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in entries:
        if entry.get("id") == instruction_id and "plan" in entry:
            plan = entry["plan"]
            return (
                {int(fdi) for fdi in plan.get("move_teeth", [])},
                {int(fdi) for fdi in plan.get("fixed_teeth", [])},
            )
    return None


def _instruction_features(text: str, fdis: list[int], case_dir: Path | None = None,
                          instruction_id: str = "i0") -> np.ndarray:
    """Convert parser output plus optional LLM move/fixed teeth into per-tooth rows."""
    cfg = parse_instruction_keywords(text)
    llm_plan = _load_llm_plan(case_dir, instruction_id) if case_dir is not None else None
    objective = cfg.objective_priority
    goal_features = np.asarray([
        any(g.get("type") == "align" for g in cfg.goals),
        any(g.get("type") == "close_spacing" for g in cfg.goals),
        any(g.get("arch") == "upper" for g in cfg.goals),
        any(g.get("arch") == "lower" for g in cfg.goals),
        any(g.get("region") == "anterior" for g in cfg.goals),
        any(g.get("region") == "posterior" for g in cfg.goals),
    ], dtype=np.float32)
    global_features = np.asarray([
        objective.get("speed", 0.0),
        objective.get("safety", 0.0),
        objective.get("aesthetics", 0.0),
        cfg.max_translation_per_stage_mm / 0.30,
        cfg.max_rotation_per_stage_deg / 2.5,
        cfg.refinements / 10.0,
        (cfg.stage_budget or 0) / 20.0,
    ], dtype=np.float32)
    rows = []
    for fdi in fdis:
        parser_protected = np.float32(fdi in cfg.protected_teeth)
        parser_movable = np.float32(cfg.movable(fdi))
        if llm_plan is not None:
            llm_move, llm_fixed = llm_plan
            llm_protected = np.float32(fdi in llm_fixed)
            llm_movable = np.float32(fdi in llm_move)
        else:
            llm_protected, llm_movable = parser_protected, parser_movable
        rows.append(np.concatenate([
            global_features,
            goal_features,
            [parser_protected],
            [parser_movable],
            [llm_protected],
            [llm_movable],
        ]))
    return np.stack(rows).astype(np.float32)


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix for an [x, y, z, w] quaternion."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _tooth_geometry(points: np.ndarray, centroid: np.ndarray, frame_q: np.ndarray,
                    fdi: int, case_center: np.ndarray, case_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create centered clouds, local-frame clouds, and stable summary features."""
    centered = (points - centroid).astype(np.float32)
    rotation = _rotation_matrix(frame_q)
    local = centered @ rotation.T
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    extent = maxs - mins
    eigenvalues = np.linalg.eigvalsh(np.cov(local, rowvar=False)).astype(np.float32)
    position = ((centroid - case_center) / case_scale).astype(np.float32)
    jaw = np.array([1.0, 0.0], dtype=np.float32) if jaw_of(fdi) == "upper" else np.array([0.0, 1.0], dtype=np.float32)
    region = np.array([1.0, 0.0], dtype=np.float32) if fdi % 10 <= 3 else np.array([0.0, 1.0], dtype=np.float32)
    fdi_features = np.array([fdi / 100.0, (fdi % 10) / 10.0], dtype=np.float32)
    geometry = np.concatenate([
        position,
        rotation.reshape(-1),
        extent,
        eigenvalues,
        jaw,
        region,
        fdi_features,
    ]).astype(np.float32)
    return centered, local, geometry

def load_sample(case_dir: str | Path, instruction: dict | None = None,
				include_targets: bool = True, augment: bool = False,
				rng: np.random.Generator | None = None) -> dict:
	"""Load one case/instruction pair into NumPy arrays and Python metadata.

	When ``augment`` is True, a small random global yaw rotation (about the case's
	vertical axis) plus isotropic point jitter is applied consistently to the input
	geometry and, when present, the gold targets — valid because rigidly rotating the
	whole scan changes nothing about which per-tooth movements are correct.
	"""
	case_dir = Path(case_dir)
	case = load_case(case_dir)
	if instruction is None:
		available = case_instructions(case_dir)
		if not available:
			raise ValueError(f"No instruction found in {case_dir}")
		instruction = available[0]

	fdis = sorted(case.teeth)
	centroids = np.stack([case.teeth[fdi].centroid for fdi in fdis])
	case_center = centroids.mean(axis=0)

	gold_path = case_dir / "gold_transforms.json"
	gold = load_gold(gold_path) if include_targets and gold_path.exists() else None

	if augment:
		rng = rng or np.random.default_rng()
		yaw_q = yaw_quat(float(rng.uniform(-np.deg2rad(8.0), np.deg2rad(8.0))))
		yaw_R = _rotation_matrix(yaw_q)
		jitter_std = 0.03  # mm; approximates scan/registration noise
		for fdi in fdis:
			tooth = case.teeth[fdi]
			tooth.centroid = yaw_R @ (tooth.centroid - case_center) + case_center
			tooth.points = ((tooth.points - case_center) @ yaw_R.T + case_center
							+ rng.normal(scale=jitter_std, size=tooth.points.shape)).astype(np.float32)
		if gold is not None:
			yaw_conj = qconj(yaw_q)
			gold = {
				fdi: (yaw_R @ np.asarray(t), qnormalize(qmul(qmul(yaw_q, q), yaw_conj)))
				for fdi, (t, q) in gold.items()
			}
		centroids = np.stack([case.teeth[fdi].centroid for fdi in fdis])
		case_center = centroids.mean(axis=0)

	case_scale = max(float(np.max(np.ptp(centroids, axis=0))), 1.0)

	points_scan, points_local, geometries = [], [], []
	for fdi in fdis:
		tooth = case.teeth[fdi]
		scan, local, geometry = _tooth_geometry(
			tooth.points, tooth.centroid, tooth.frame_q, fdi, case_center, case_scale
		)
		points_scan.append(scan)
		points_local.append(local)
		geometries.append(geometry)

	sample = {
		"case_id": case.case_id,
		"instruction_id": instruction.get("id", "i0"),
		"instruction": instruction.get("text", ""),
		"fdis": np.asarray(fdis, dtype=np.int32),
		"points": np.stack(points_scan),
		"points_local": np.stack(points_local),
		"geometry": np.stack(geometries),
		"centroids_mm": centroids.astype(np.float32),
		"instruction_features": _instruction_features(
			instruction.get("text", ""), fdis, case_dir, instruction.get("id", "i0")
		),
		"meta": dict(case.meta),
		"meta_features": np.asarray([
			1.0 if case.meta.get("arch_type") == "upper" else 0.0,
			1.0 if case.meta.get("arch_type") == "lower" else 0.0,
			1.0 if case.meta.get("arch_type") == "dual" else 0.0,
			{"Simple": 0.0, "Moderate": 0.5, "Complex": 1.0}.get(case.meta.get("complexity"), 0.0),
		], dtype=np.float32),
	}

	if gold is not None:
		translations, rotations, target_mask, protected_mask = [], [], [], []
		for fdi in fdis:
			if fdi in gold:
				translation, rotation = gold[fdi]
				target_mask.append(1.0)
				held = (np.linalg.norm(translation) < GOLD_ZERO_TOL_MM
						and geodesic_deg(rotation, IDENTITY_Q) < GOLD_ZERO_TOL_DEG)
				protected_mask.append(1.0 if held else 0.0)
			else:
				translation, rotation = np.zeros(3), IDENTITY_Q
				target_mask.append(0.0)
				protected_mask.append(0.0)
			translations.append(translation)
			rotations.append(rotation)
		sample["target_translation"] = np.asarray(translations, dtype=np.float32)
		sample["target_rotation"] = np.asarray(rotations, dtype=np.float32)
		sample["target_mask"] = np.asarray(target_mask, dtype=np.float32)
		sample["protected_mask"] = np.asarray(protected_mask, dtype=np.float32)

	return sample

@dataclass(frozen=True)
class RichFeatureConfig:
    same_arch_neighbors: int = 4
    opposite_arch_neighbors: int = 4
    arch_polynomial_degree: int = 2
    include_surface_distance: bool = True
    use_llm_stage1: bool = True


def _safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(x * x, axis=axis, keepdims=keepdims), EPS))


def _unit(x: np.ndarray) -> np.ndarray:
    return x / _safe_norm(x, axis=-1, keepdims=True)


def _pairwise_dist(x: np.ndarray) -> np.ndarray:
    delta = x[:, None, :] - x[None, :, :]
    return _safe_norm(delta, axis=-1)


def _jaw_masks(fdis: np.ndarray, geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use geometry jaw one-hot when present; otherwise infer from FDI quadrant."""
    if geometry.ndim == 2 and geometry.shape[1] >= 20:
        upper = geometry[:, 18] >= geometry[:, 19]
    else:
        quadrant = fdis.astype(np.int64) // 10
        upper = np.isin(quadrant, [1, 2])
    return upper, ~upper


def _rotation_matrices(geometry: np.ndarray) -> np.ndarray:
    """Extract the 3x3 current tooth frames from geometry columns 3:12."""
    if geometry.shape[1] < 12:
        return np.repeat(np.eye(3, dtype=np.float32)[None], len(geometry), axis=0)
    return geometry[:, 3:12].reshape(-1, 3, 3).astype(np.float32)


def _centroids_and_scale(sample: dict[str, Any]) -> tuple[np.ndarray, float, str]:
    """Recover centroids in millimetres when metadata exposes normalization.

    The original geometry stores normalized centroid position in columns 0:3.
    If ``meta.case_scale`` is unavailable, relational centroid distances remain
    in normalized scan units and ``relational_distance_unit`` reports that fact.
    """
    if "centroids_mm" in sample:
        centroids = np.asarray(sample["centroids_mm"], dtype=np.float32)
        return centroids, 1.0, "mm"

    pos = np.asarray(sample["geometry"][:, :3], dtype=np.float32)
    meta = sample.get("meta") or {}
    if isinstance(meta, dict):
        scale = meta.get("case_scale", meta.get("scale"))
        center = meta.get("case_center", meta.get("center"))
        if scale is not None:
            scale = float(np.asarray(scale).reshape(-1)[0])
            if np.isfinite(scale) and scale > EPS:
                if center is None:
                    center_arr = np.zeros(3, dtype=np.float32)
                else:
                    center_arr = np.asarray(center, dtype=np.float32).reshape(3)
                return pos * scale + center_arr, scale, "mm"
    return pos, 1.0, "normalized_scan_unit"


def _global_clouds(sample: dict[str, Any], centroids: np.ndarray, scale: float,
                   distance_unit: str) -> np.ndarray:
    """Reconstruct scan-frame clouds in the same units as centroids."""
    centered = np.asarray(sample["points"], dtype=np.float32)
    if distance_unit == "mm":
        return centered + centroids[:, None, :]
    # centered point clouds are in source scan units while positions are
    # normalized. Convert centered offsets using the known normalization scale;
    # when scale is unavailable (=1), this is only an approximation.
    return centered / max(scale, EPS) + centroids[:, None, :]


def _minimum_surface_distance(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> float:
    """Exact minimum point-to-point distance with bounded temporary memory."""
    best = np.inf
    for start in range(0, len(a), chunk):
        delta = a[start:start + chunk, None, :] - b[None, :, :]
        best = min(best, float(np.sqrt(np.min(np.sum(delta * delta, axis=-1)))))
    return best


def _fit_arch_features(centroids: np.ndarray, mask: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2-D polynomial centerline in the arch PCA plane.

    Returns signed centreline offset, orientation-vs-tangent features
    [cos(theta), sin(theta)], and normalized location along the arch.
    """
    n = len(centroids)
    offset = np.zeros(n, dtype=np.float32)
    tangent = np.zeros((n, 2), dtype=np.float32)
    arc_position = np.zeros(n, dtype=np.float32)
    ids = np.flatnonzero(mask)
    if len(ids) < 3:
        tangent[ids, 0] = 1.0
        return offset, tangent, arc_position

    xyz = centroids[ids]
    origin = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - origin, full_matrices=False)
    plane = vt[:2].T
    uv = (xyz - origin) @ plane
    # Choose the PCA coordinate with greater range as the curve independent axis.
    if np.ptp(uv[:, 1]) > np.ptp(uv[:, 0]):
        uv = uv[:, ::-1]
    x, y = uv[:, 0], uv[:, 1]
    deg = int(min(max(1, degree), len(ids) - 1))
    coef = np.polyfit(x, y, deg=deg)
    fitted = np.polyval(coef, x)
    deriv = np.polyval(np.polyder(coef), x)
    denom = np.sqrt(1.0 + deriv * deriv)
    offset[ids] = ((y - fitted) / denom).astype(np.float32)
    tangent[ids, 0] = (1.0 / denom).astype(np.float32)
    tangent[ids, 1] = (deriv / denom).astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    arc_position[ids] = ((x - xmin) / max(xmax - xmin, EPS)).astype(np.float32)
    return offset, tangent, arc_position


def _local_space_features(centroids: np.ndarray, extents: np.ndarray,
                          upper: np.ndarray) -> np.ndarray:
    """Approximate mesiodistal available space and crowding from nearest same-arch teeth.

    Columns: left distance, right distance, available space, estimated tooth width,
    and signed space surplus (positive=space, negative=possible crowding).
    """
    out = np.zeros((len(centroids), 5), dtype=np.float32)
    for jaw_mask in (upper, ~upper):
        ids = np.flatnonzero(jaw_mask)
        if len(ids) < 2:
            continue
        xyz = centroids[ids]
        _, _, vt = np.linalg.svd(xyz - xyz.mean(axis=0), full_matrices=False)
        order = np.argsort((xyz - xyz.mean(axis=0)) @ vt[0])
        ordered = ids[order]
        for rank, i in enumerate(ordered):
            left = float(_safe_norm(centroids[i] - centroids[ordered[rank - 1]])) if rank else 0.0
            right = float(_safe_norm(centroids[i] - centroids[ordered[rank + 1]])) if rank + 1 < len(ordered) else 0.0
            available = 0.5 * (left + right)
            width = float(np.max(extents[i]))
            out[i] = [left, right, available, width, available - width]
    return out


def build_rich_features(sample: dict[str, Any], config: RichFeatureConfig = RichFeatureConfig()) -> dict[str, Any]:
    """Add tooth, pairwise, neighbourhood, arch, and cross-arch features."""
    result = dict(sample)
    fdis = np.asarray(sample["fdis"], dtype=np.int64)
    geometry = np.asarray(sample["geometry"], dtype=np.float32)
    instruction = np.asarray(sample["instruction_features"], dtype=np.float32).copy()
    if not config.use_llm_stage1 and instruction.shape[1] >= 17:
        instruction[:, 15:17] = 0.0
    t = len(fdis)
    centroids, case_scale, distance_unit = _centroids_and_scale(sample)
    rotations = _rotation_matrices(geometry)
    upper, lower = _jaw_masks(fdis, geometry)
    jaw_code = np.where(upper, 0, 1).astype(np.int64)
    extents = geometry[:, 12:15] if geometry.shape[1] >= 15 else np.zeros((t, 3), np.float32)

    # Pairwise directed relations i -> j.
    delta = centroids[None, :, :] - centroids[:, None, :]
    distance = _safe_norm(delta, axis=-1)
    direction = delta / distance[..., None]
    rel_rot = np.einsum("tji,sjk->tsik", rotations, rotations)
    rel_rot_6d = rel_rot[..., :, :2].reshape(t, t, 6)
    same_arch = (jaw_code[:, None] == jaw_code[None, :]).astype(np.float32)
    opposite_arch = 1.0 - same_arch
    same_tooth = np.eye(t, dtype=np.float32)

    global_clouds = _global_clouds(sample, centroids, case_scale, distance_unit)
    surface_distance = np.zeros((t, t), dtype=np.float32)
    if config.include_surface_distance:
        for i in range(t):
            for j in range(i + 1, t):
                d = _minimum_surface_distance(global_clouds[i], global_clouds[j])
                surface_distance[i, j] = surface_distance[j, i] = d

    pair_features = np.concatenate([
        delta.astype(np.float32),
        distance[..., None].astype(np.float32),
        direction.astype(np.float32),
        rel_rot_6d.astype(np.float32),
        surface_distance[..., None],
        same_arch[..., None],
        opposite_arch[..., None],
        same_tooth[..., None],
    ], axis=-1).astype(np.float32)

    # Fixed-K neighbour tensors and masks.
    k_same = config.same_arch_neighbors
    k_opp = config.opposite_arch_neighbors
    same_idx = np.full((t, k_same), -1, dtype=np.int64)
    opp_idx = np.full((t, k_opp), -1, dtype=np.int64)
    same_mask = np.zeros((t, k_same), dtype=np.float32)
    opp_mask = np.zeros((t, k_opp), dtype=np.float32)
    for i in range(t):
        candidates_same = np.flatnonzero((jaw_code == jaw_code[i]) & (np.arange(t) != i))
        candidates_opp = np.flatnonzero(jaw_code != jaw_code[i])
        candidates_same = candidates_same[np.argsort(distance[i, candidates_same])][:k_same]
        candidates_opp = candidates_opp[np.argsort(distance[i, candidates_opp])][:k_opp]
        same_idx[i, :len(candidates_same)] = candidates_same
        opp_idx[i, :len(candidates_opp)] = candidates_opp
        same_mask[i, :len(candidates_same)] = 1.0
        opp_mask[i, :len(candidates_opp)] = 1.0

    safe_same = np.maximum(same_idx, 0)
    safe_opp = np.maximum(opp_idx, 0)
    same_features = pair_features[np.arange(t)[:, None], safe_same] * same_mask[..., None]
    opp_features = pair_features[np.arange(t)[:, None], safe_opp] * opp_mask[..., None]

    upper_offset, upper_tangent, upper_arc_pos = _fit_arch_features(
        centroids, upper, config.arch_polynomial_degree)
    lower_offset, lower_tangent, lower_arc_pos = _fit_arch_features(
        centroids, lower, config.arch_polynomial_degree)
    arch_offset = upper_offset + lower_offset
    arch_tangent = upper_tangent + lower_tangent
    arch_position = upper_arc_pos + lower_arc_pos
    local_space = _local_space_features(centroids, extents, upper)

    # Contralateral partner by mirrored FDI quadrant when available.
    fdi_to_idx = {int(f): i for i, f in enumerate(fdis)}
    contra_idx = np.full(t, -1, dtype=np.int64)
    contra_features = np.zeros((t, pair_features.shape[-1]), dtype=np.float32)
    for i, fdi in enumerate(fdis):
        q, tooth = int(fdi) // 10, int(fdi) % 10
        cq = {1: 2, 2: 1, 3: 4, 4: 3}.get(q)
        if cq is not None and 10 * cq + tooth in fdi_to_idx:
            j = fdi_to_idx[10 * cq + tooth]
            contra_idx[i] = j
            contra_features[i] = pair_features[i, j]

    # Explicit tooth-level features. Keep shape clouds separate for PointNet/PointTransformer.
    tooth_features = np.concatenate([
        geometry,
        instruction,
        arch_offset[:, None],
        arch_tangent,
        arch_position[:, None],
        local_space,
        contra_features,
    ], axis=-1).astype(np.float32)

    result.update({
        "tooth_features": tooth_features,
        "centroids": centroids.astype(np.float32),
        "current_rotations": rotations,
        "jaw_index": jaw_code,
        "tooth_mask": np.ones(t, dtype=np.float32),
        "pair_features": pair_features,
        "pair_mask": (1.0 - same_tooth).astype(np.float32),
        "same_arch_neighbor_index": same_idx,
        "same_arch_neighbor_features": same_features.astype(np.float32),
        "same_arch_neighbor_mask": same_mask,
        "opposite_arch_neighbor_index": opp_idx,
        "opposite_arch_neighbor_features": opp_features.astype(np.float32),
        "opposite_arch_neighbor_mask": opp_mask,
        "contralateral_index": contra_idx,
        "arch_offset": arch_offset.astype(np.float32),
        "arch_tangent": arch_tangent.astype(np.float32),
        "arch_position": arch_position.astype(np.float32),
        "local_space_features": local_space,
        "llm_stage1_move": instruction[:, 16:17] if instruction.shape[1] >= 17 else np.zeros((t, 1), dtype=np.float32),
        "llm_stage1_fixed": instruction[:, 15:16] if instruction.shape[1] >= 17 else np.zeros((t, 1), dtype=np.float32),
    })
    meta = dict(result.get("meta") or {})
    meta.update({
        "relational_distance_unit": distance_unit,
        "rich_tooth_feature_dim": int(tooth_features.shape[-1]),
        "pair_feature_dim": int(pair_features.shape[-1]),
    })
    result["meta"] = meta
    return result


class RichTreatmentPlanDataset:
    """Dataset over every case/instruction pair in a split."""

    def __init__(self, pack_dir: str | Path, split: str = "train",
                 include_targets: bool | None = None, augment: bool = False,
                 config: RichFeatureConfig = RichFeatureConfig(), seed: int = 0):
        if include_targets is None:
            include_targets = split == "train"
        self.pack_dir = Path(pack_dir)
        self.split = split
        self.include_targets = include_targets
        self.augment = augment
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.samples: list[tuple[Path, dict[str, Any]]] = []
        for case_dir in sorted((self.pack_dir / split).iterdir()):
            if case_dir.is_dir():
                for instruction in case_instructions(case_dir):
                    self.samples.append((case_dir, instruction))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_dir, instruction = self.samples[index]
        sample = load_sample(
            case_dir,
            instruction=instruction,
            include_targets=self.include_targets,
            augment=self.augment,
            rng=self.rng,
        )
        return build_rich_features(sample, self.config)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


def _pad_first_axis(x: np.ndarray, size: int, value: float | int = 0) -> np.ndarray:
    shape = (size,) + x.shape[1:]
    out = np.full(shape, value, dtype=x.dtype)
    out[:len(x)] = x
    return out


def _pad_pair_axes(x: np.ndarray, size: int, value: float | int = 0) -> np.ndarray:
    shape = (size, size) + x.shape[2:]
    out = np.full(shape, value, dtype=x.dtype)
    out[:x.shape[0], :x.shape[1]] = x
    return out


def collate_rich_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable tooth counts and stack a rich batch.

    Arrays beginning with (T,T,...) are padded on both tooth axes. Arrays
    beginning with (T,...) are padded on the first axis. Other numeric arrays
    are stacked only when their shapes agree.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    max_teeth = max(len(s["fdis"]) for s in samples)
    result: dict[str, Any] = {}
    metadata_keys = {"case_id", "instruction_id", "instruction", "meta"}
    all_keys = set().union(*(s.keys() for s in samples))
    for key in sorted(all_keys):
        values = [s.get(key) for s in samples]
        if key in metadata_keys or not all(isinstance(v, np.ndarray) for v in values):
            result[key] = values
            continue
        arrays = values
        is_pair = all(a.ndim >= 2 and a.shape[0] == len(s["fdis"]) and
                      a.shape[1] == len(s["fdis"]) for a, s in zip(arrays, samples))
        is_tooth = all(a.ndim >= 1 and a.shape[0] == len(s["fdis"])
                       for a, s in zip(arrays, samples))
        fill = -1 if key.endswith("index") or key == "fdis" else 0
        if is_pair:
            result[key] = np.stack([_pad_pair_axes(a, max_teeth, fill) for a in arrays])
        elif is_tooth:
            result[key] = np.stack([_pad_first_axis(a, max_teeth, fill) for a in arrays])
        elif len({a.shape for a in arrays}) == 1:
            result[key] = np.stack(arrays)
        else:
            result[key] = arrays
    result["batch_tooth_mask"] = np.stack([
        np.r_[np.ones(len(s["fdis"]), np.float32),
              np.zeros(max_teeth - len(s["fdis"]), np.float32)]
        for s in samples
    ])
    return result


def make_torch_loader(pack_dir: str | Path, split: str = "train",
                      batch_size: int = 4, shuffle: bool = False,
                      num_workers: int = 0,
                      config: RichFeatureConfig = RichFeatureConfig()):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("make_torch_loader requires PyTorch") from exc
    dataset = RichTreatmentPlanDataset(pack_dir, split=split, config=config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_rich_samples)


def smoke_test(pack_dir: str | Path, split: str = "train") -> None:
    dataset = RichTreatmentPlanDataset(pack_dir, split=split)
    sample = dataset[0]
    t = len(sample["fdis"])
    assert sample["tooth_features"].shape[0] == t
    assert sample["pair_features"].shape[:2] == (t, t)
    assert sample["points_local"].shape[0] == t
    assert np.isfinite(sample["tooth_features"]).all()
    assert np.isfinite(sample["pair_features"]).all()
    print({
        "case_id": sample["case_id"],
        "teeth": t,
        "points_local": sample["points_local"].shape,
        "tooth_features": sample["tooth_features"].shape,
        "pair_features": sample["pair_features"].shape,
        "distance_unit": sample["meta"]["relational_distance_unit"],
        "has_targets": "target_translation" in sample,
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smoke-test rich orthodontic dataloader")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    args = parser.parse_args()
    smoke_test(args.pack, args.split)
