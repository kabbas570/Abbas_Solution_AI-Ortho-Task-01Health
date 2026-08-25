"""The task I/O contract — case format, plan format, instruction config, math helpers.

Ships verbatim in the candidate starter kit, so: numpy + stdlib only, no repo imports.

Conventions (the pack README §2 is the prose version):
- Units: millimetres. Quaternions: [x, y, z, w], unit norm.
- A plan transform is absolute-from-initial:  x' = R(q) @ (x - c) + c + t
  where c is the tooth centroid at the *start* position.
- FDI numbering. The pack carries 28 teeth (no third molars): upper 17..27, lower 47..37.
- Per-tooth local frame `frame_q` rotates a vector FROM the scan frame INTO the tooth frame:
  v_tooth = R(frame_q) @ v_scan.  Tooth axes: x = mesiodistal (toward distal),
  y = buccolingual (outward), z = occluso-apical (up).

Case directory layout (train; eval cases omit gold_* and movements.csv):
    case_<id>/
      meta.json             {"case_id", "arch_type", "complexity", "units": "mm"}
      teeth.json            {"<fdi>": {"centroid": [3], "frame_q": [4]}}
      points.npz            arrays keyed "t<fdi>", each (N, 3) float32, scan frame, mm
      instruction.txt       one free-text instruction        (train)
      instructions.json     [{"id", "text"}, ...]            (eval battery)
      gold_transforms.json  {"<fdi>": {"t_mm": [3], "q": [4]}} (train)
      movements.csv         clinical components per tooth     (train)

Plan JSON (one file per (case, instruction)):
    {"format": "taskplan-1", "case_id": "...", "instruction_id": "...",
     "instruction": "...", "transforms": {"<fdi>": {"t_mm": [3], "q": [4]}},
     "meta": {...}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── FDI vocabulary ────────────────────────────────────────────────────────────

UPPER_FDI = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
LOWER_FDI = [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37]
ALL_FDI = UPPER_FDI + LOWER_FDI

# Canonical mesiodistal crown widths (mm) by tooth position (unit digit), per jaw.
_WIDTHS_UPPER = {1: 8.5, 2: 6.5, 3: 7.6, 4: 7.1, 5: 6.6, 6: 10.4, 7: 9.8}
_WIDTHS_LOWER = {1: 5.3, 2: 5.9, 3: 6.8, 4: 7.0, 5: 7.1, 6: 11.4, 7: 10.5}


def jaw_of(fdi: int) -> str:
    return "upper" if fdi // 10 in (1, 2) else "lower"


def region_of(fdi: int) -> str:
    return "anterior" if fdi % 10 <= 3 else "posterior"


def width_of(fdi: int) -> float:
    table = _WIDTHS_UPPER if jaw_of(fdi) == "upper" else _WIDTHS_LOWER
    return table[fdi % 10]


# ── Quaternion helpers ([x, y, z, w]) ─────────────────────────────────────────

def qnormalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate points v (…, 3) by unit quaternion q."""
    q = qnormalize(q)
    u, w = q[:3], q[3]
    v = np.asarray(v, dtype=np.float64)
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def yaw_quat(angle_rad: float) -> np.ndarray:
    """Rotation about global +z by angle_rad."""
    return np.array([0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)])


def geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """Angle (degrees) of the relative rotation between two unit quaternions."""
    d = abs(float(np.dot(qnormalize(qa), qnormalize(qb))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


# ── Case / plan structures ────────────────────────────────────────────────────

@dataclass
class Tooth:
    fdi: int
    points: np.ndarray          # (N, 3) scan frame, mm, at the START position
    centroid: np.ndarray        # (3,)
    frame_q: np.ndarray         # (4,) scan→tooth rotation

    def local(self, v: np.ndarray) -> np.ndarray:
        """Express a scan-frame vector in this tooth's local frame."""
        return qrot(self.frame_q, v)


@dataclass
class Case:
    case_id: str
    teeth: dict[int, Tooth]
    meta: dict = field(default_factory=dict)

    @property
    def fdis(self) -> list[int]:
        return sorted(self.teeth.keys())


@dataclass
class Plan:
    case_id: str
    instruction_id: str
    instruction: str
    # fdi -> (t_mm (3,), q (4,))
    transforms: dict[int, tuple[np.ndarray, np.ndarray]]
    meta: dict = field(default_factory=dict)

    def movement_mm(self, fdi: int) -> float:
        t, _ = self.transforms[fdi]
        return float(np.linalg.norm(t))

    def rotation_deg(self, fdi: int) -> float:
        _, q = self.transforms[fdi]
        return geodesic_deg(q, IDENTITY_Q)


def apply_transform(tooth: Tooth, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the tooth's points at the planned position: x' = R(q)(x − c) + c + t."""
    return qrot(q, tooth.points - tooth.centroid) + tooth.centroid + t


# ── Instruction config (the parsed contract between text and geometry) ───────

DEFAULT_CAPS = {"max_translation_per_stage_mm": 0.25, "max_rotation_per_stage_deg": 2.0}
SAFE_CAPS = {"max_translation_per_stage_mm": 0.15, "max_rotation_per_stage_deg": 1.0}
FAST_CAPS = {"max_translation_per_stage_mm": 0.30, "max_rotation_per_stage_deg": 2.5}


@dataclass
class InstructionConfig:
    """Versioned schema; the single handle text has on geometry (spec §3/§4)."""

    objective_priority: dict = field(
        default_factory=lambda: {"speed": 0.33, "safety": 0.34, "aesthetics": 0.33}
    )
    stage_budget: int | None = None
    max_translation_per_stage_mm: float = DEFAULT_CAPS["max_translation_per_stage_mm"]
    max_rotation_per_stage_deg: float = DEFAULT_CAPS["max_rotation_per_stage_deg"]
    refinements: int = 0
    protected_teeth: list[int] = field(default_factory=list)
    # goals: [{"type": "align", "arch": "upper|lower|both", "region": "anterior|posterior|full"}]
    goals: list[dict] = field(default_factory=lambda: [{"type": "align", "arch": "both", "region": "full"}])
    version: str = "config-1"

    def movable(self, fdi: int) -> bool:
        """A tooth may move iff it is not protected AND some goal covers it."""
        if fdi in self.protected_teeth:
            return False
        for g in self.goals:
            arch_ok = g.get("arch", "both") in ("both", jaw_of(fdi))
            region_ok = g.get("region", "full") in ("full", region_of(fdi))
            if arch_ok and region_ok:
                return True
        return False

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "objective_priority": self.objective_priority,
            "stage_budget": self.stage_budget,
            "max_translation_per_stage_mm": self.max_translation_per_stage_mm,
            "max_rotation_per_stage_deg": self.max_rotation_per_stage_deg,
            "refinements": self.refinements,
            "protected_teeth": sorted(self.protected_teeth),
            "goals": self.goals,
        }

    @classmethod
    def from_json(cls, d: dict) -> "InstructionConfig":
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kw)


# ── Disk I/O ──────────────────────────────────────────────────────────────────

def load_case(case_dir: str | Path) -> Case:
    case_dir = Path(case_dir)
    meta = json.loads((case_dir / "meta.json").read_text())
    teeth_meta = json.loads((case_dir / "teeth.json").read_text())
    npz = np.load(case_dir / "points.npz")
    teeth: dict[int, Tooth] = {}
    for fdi_s, tm in teeth_meta.items():
        fdi = int(fdi_s)
        teeth[fdi] = Tooth(
            fdi=fdi,
            points=np.asarray(npz[f"t{fdi}"], dtype=np.float64),
            centroid=np.asarray(tm["centroid"], dtype=np.float64),
            frame_q=qnormalize(np.asarray(tm["frame_q"], dtype=np.float64)),
        )
    return Case(case_id=meta["case_id"], teeth=teeth, meta=meta)


def case_instructions(case_dir: str | Path) -> list[dict]:
    """Instructions to plan for: eval battery file, or the single train instruction."""
    case_dir = Path(case_dir)
    battery = case_dir / "instructions.json"
    if battery.exists():
        return json.loads(battery.read_text())
    single = case_dir / "instruction.txt"
    return [{"id": "i0", "text": single.read_text(encoding="utf-8").strip()}] if single.exists() else []


def load_gold(path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    d = json.loads(Path(path).read_text())
    return {
        int(fdi): (np.asarray(v["t_mm"], dtype=np.float64), qnormalize(np.asarray(v["q"])))
        for fdi, v in d["transforms"].items()
    }


def save_plan(plan: Plan, path: str | Path) -> None:
    payload = {
        "format": "taskplan-1",
        "case_id": plan.case_id,
        "instruction_id": plan.instruction_id,
        "instruction": plan.instruction,
        "transforms": {
            str(fdi): {"t_mm": np.asarray(t).round(5).tolist(), "q": qnormalize(q).round(6).tolist()}
            for fdi, (t, q) in plan.transforms.items()
        },
        "meta": plan.meta,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))


def load_plan(path: str | Path) -> Plan:
    d = json.loads(Path(path).read_text())
    if d.get("format") != "taskplan-1":
        raise ValueError(f"{path}: unknown plan format {d.get('format')!r}")
    return Plan(
        case_id=d["case_id"],
        instruction_id=d.get("instruction_id", "i0"),
        instruction=d.get("instruction", ""),
        transforms={
            int(fdi): (np.asarray(v["t_mm"], dtype=np.float64), qnormalize(np.asarray(v["q"])))
            for fdi, v in d["transforms"].items()
        },
        meta=d.get("meta", {}),
    )


def plan_filename(case_id: str, instruction_id: str) -> str:
    return f"{case_id}__{instruction_id}.json"
