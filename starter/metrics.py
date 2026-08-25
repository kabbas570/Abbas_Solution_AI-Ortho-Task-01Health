"""The task scoring harness — the same code path for self-eval (public) and judging (frozen).

Three metric families (pack README §6 is the prose version):

1. endpoint       — per-tooth translation error (mm) and rotation error ( deg) vs gold, with
                    "% teeth within 0.5 mm AND 2 deg" as the headline accuracy figure.
2. constraints    — immobility compliance on teeth the gold holds fixed (protected /
                    out-of-scope), plus an interpenetration proxy between adjacent teeth
                    (axis-projected extent overlap — cheap, orientation-aware, monotone in
                    true overlap; note it has a nonzero floor on real contacting dentitions).
3. sensitivity    — instruction counterfactuals on the eval battery: does the *prediction*
                    change between instructions the way the *gold* changes, in magnitude
                    (ratio) and direction (cosine)? A text-blind model scores ~ 0 on both.

Scores are pure functions of (pack gold, plans dir): no network, no DB, numpy only.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from contract import (
    IDENTITY_Q,
    Case,
    apply_transform,
    case_instructions,
    geodesic_deg,
    jaw_of,
    load_case,
    load_gold,
    load_plan,
    plan_filename,
    width_of,
)

TRANS_TOL_MM = 0.5
ROT_TOL_DEG = 2.0
IMMOBILE_TOL_MM = 0.25
IMMOBILE_TOL_DEG = 2.0
CONTACT_TOL_MM = 0.05  # penetration deeper than this counts as a collision


# -- per-(case, instruction) scoring ------------------------------------------

def _endpoint_errors(pred: dict, gold: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    fdis = sorted(set(pred.keys()) & set(gold.keys()))
    et = np.array([np.linalg.norm(pred[f][0] - gold[f][0]) for f in fdis])
    er = np.array([geodesic_deg(pred[f][1], gold[f][1]) for f in fdis])
    return et, er, fdis


def _immobile_violations(pred: dict, gold: dict) -> tuple[int, int]:
    """Among teeth gold holds fixed, count predictions that move them anyway."""
    held = [f for f, (t, q) in gold.items()
            if np.linalg.norm(t) < 1e-6 and geodesic_deg(q, IDENTITY_Q) < 1e-6]
    bad = 0
    for f in held:
        if f not in pred:
            continue
        t, q = pred[f]
        if np.linalg.norm(t) > IMMOBILE_TOL_MM or geodesic_deg(q, IDENTITY_Q) > IMMOBILE_TOL_DEG:
            bad += 1
    return bad, len(held)


def _penetration_proxy(case: Case, pred: dict) -> tuple[float, int]:
    """Max axis-projected overlap (mm) between adjacent same-jaw teeth after the plan.

    For each adjacent pair, project both point clouds on the centroid axis; overlap of the
    projected extents approximates interpenetration depth. Cheap, orientation-aware, and
    monotone in true overlap — sufficient for ranking submissions.
    """
    worst, n_colliding = 0.0, 0
    by_jaw: dict[str, list[int]] = {"upper": [], "lower": []}
    for f in case.fdis:
        by_jaw[jaw_of(f)].append(f)
    # adjacency = arch order, recovered by angle around the jaw centroid
    for jaw, fdis in by_jaw.items():
        if len(fdis) < 2:
            continue
        cents = {f: case.teeth[f].centroid for f in fdis}
        mid = np.mean([c for c in cents.values()], axis=0)
        order = sorted(fdis, key=lambda f: np.arctan2(*(cents[f][:2] - mid[:2])[::-1]))
        moved = {}
        for f in order:
            t, q = pred.get(f, (np.zeros(3), IDENTITY_Q))
            moved[f] = apply_transform(case.teeth[f], np.asarray(t), np.asarray(q))
        for f1, f2 in zip(order, order[1:]):
            c1, c2 = moved[f1].mean(axis=0), moved[f2].mean(axis=0)
            axis = c2 - c1
            d = np.linalg.norm(axis)
            if d < 1e-9:
                continue
            u = axis / d
            # 97th-percentile extent instead of max: robust to superellipsoid corner
            # overshoot, still monotone in true overlap
            ext1 = float(np.percentile((moved[f1] - c1) @ u, 97))
            ext2 = float(np.percentile((c2 - moved[f2]) @ u, 97))
            pen = ext1 + ext2 - d
            if pen > CONTACT_TOL_MM:
                n_colliding += 1
                worst = max(worst, pen)
    return worst, n_colliding


# -- battery-level sensitivity ------------------------------------------------

def _stack_delta(a: dict, b: dict) -> np.ndarray:
    fdis = sorted(set(a.keys()) & set(b.keys()))
    return np.concatenate([np.asarray(a[f][0]) - np.asarray(b[f][0]) for f in fdis])


def _sensitivity(case_plans: dict[str, dict], case_golds: dict[str, dict]) -> dict | None:
    """Counterfactual agreement across the instruction battery of one case."""
    iids = sorted(set(case_plans.keys()) & set(case_golds.keys()))
    if len(iids) < 2:
        return None
    ratios, cosines = [], []
    for i, j in combinations(iids, 2):
        dp = _stack_delta(case_plans[i], case_plans[j])
        dg = _stack_delta(case_golds[i], case_golds[j])
        gnorm = float(np.linalg.norm(dg))
        pnorm = float(np.linalg.norm(dp))
        if gnorm < 1e-6:
            continue  # instructions happen to imply the same gold — uninformative pair
        ratios.append(pnorm / gnorm)
        cosines.append(float(dp @ dg / (pnorm * gnorm)) if pnorm > 1e-9 else 0.0)
    if not ratios:
        return None
    return {
        "pairs": len(ratios),
        "magnitude_ratio": float(np.mean(ratios)),      # 1.0 = deltas as large as gold's
        "direction_cosine": float(np.mean(cosines)),    # 1.0 = deltas point the right way
    }


# -- the harness --------------------------------------------------------------

def score_pack(pack_dir: str | Path, plans_dir: str | Path,
               split: str = "eval") -> dict:
    """Score every plan in plans_dir against the pack's gold for `split`.

    For split="eval", gold lives in pack/gold/ (the judge's copy). For split="train",
    gold_transforms.json inside each case dir is used (candidate self-eval).
    """
    pack, plans = Path(pack_dir), Path(plans_dir)
    case_dirs = sorted((pack / split).iterdir())

    per_pair = []
    all_et, all_er = [], []
    imm_bad = imm_total = 0
    pen_worst, pen_pairs = 0.0, 0
    missing = []
    sens_per_case = []

    for cdir in case_dirs:
        if not cdir.is_dir():
            continue
        case = load_case(cdir)
        battery = case_instructions(cdir)
        case_plans, case_golds = {}, {}
        for instr in battery:
            iid = instr["id"]
            gold_path = (
                pack / "gold" / plan_filename(case.case_id, iid)
                if split == "eval" else cdir / "gold_transforms.json"
            )
            if not gold_path.exists():
                continue
            gold = load_gold(gold_path)
            plan_path = plans / plan_filename(case.case_id, iid)
            if not plan_path.exists():
                missing.append(plan_path.name)
                continue
            plan = load_plan(plan_path)
            pred = plan.transforms

            et, er, _ = _endpoint_errors(pred, gold)
            all_et.append(et)
            all_er.append(er)
            bad, tot = _immobile_violations(pred, gold)
            imm_bad += bad
            imm_total += tot
            pw, pc = _penetration_proxy(case, pred)
            pen_worst = max(pen_worst, pw)
            pen_pairs += pc

            case_plans[iid], case_golds[iid] = pred, gold
            per_pair.append({
                "case_id": case.case_id, "instruction_id": iid,
                "trans_err_mean_mm": float(et.mean()) if len(et) else None,
                "rot_err_mean_deg": float(er.mean()) if len(er) else None,
                "pct_within_tol": float(np.mean((et <= TRANS_TOL_MM) & (er <= ROT_TOL_DEG))) if len(et) else None,
                "immobile_violations": bad, "immobile_total": tot,
                "worst_penetration_mm": round(pw, 3), "colliding_pairs": pc,
            })
        s = _sensitivity(case_plans, case_golds)
        if s is not None:
            sens_per_case.append(s)

    et = np.concatenate(all_et) if all_et else np.array([])
    er = np.concatenate(all_er) if all_er else np.array([])
    report = {
        "split": split,
        "n_plans_scored": len(per_pair),
        "n_plans_missing": len(missing),
        "missing": missing[:20],
        "endpoint": {
            "trans_err_mean_mm": float(et.mean()) if len(et) else None,
            "trans_err_median_mm": float(np.median(et)) if len(et) else None,
            "trans_err_p90_mm": float(np.percentile(et, 90)) if len(et) else None,
            "rot_err_mean_deg": float(er.mean()) if len(er) else None,
            "rot_err_median_deg": float(np.median(er)) if len(er) else None,
            "pct_teeth_within_0p5mm_2deg": float(np.mean((et <= TRANS_TOL_MM) & (er <= ROT_TOL_DEG))) if len(et) else None,
        },
        "constraints": {
            "immobility_violation_rate": (imm_bad / imm_total) if imm_total else None,
            "immobile_teeth_checked": imm_total,
            "worst_penetration_mm": round(pen_worst, 3),
            "colliding_adjacent_pairs": pen_pairs,
        },
        "sensitivity": {
            "cases_with_battery": len(sens_per_case),
            "magnitude_ratio_mean": float(np.mean([s["magnitude_ratio"] for s in sens_per_case])) if sens_per_case else None,
            "direction_cosine_mean": float(np.mean([s["direction_cosine"] for s in sens_per_case])) if sens_per_case else None,
        },
        "per_pair": per_pair,
    }
    return report


def format_report(report: dict, name: str = "") -> str:
    e, c, s = report["endpoint"], report["constraints"], report["sensitivity"]

    def fmt(v, spec=".3f"):
        return "-" if v is None else format(v, spec)  # ASCII: survives cp1252 redirects

    lines = [
        f"-- task score{': ' + name if name else ''} ({report['split']}, "
        f"{report['n_plans_scored']} plans, {report['n_plans_missing']} missing) --",
        f"endpoint     trans err mean {fmt(e['trans_err_mean_mm'])} mm | median {fmt(e['trans_err_median_mm'])} | p90 {fmt(e['trans_err_p90_mm'])}",
        f"             rot err mean {fmt(e['rot_err_mean_deg'], '.2f')} deg | within-tol (<=0.5mm & <=2 deg) {fmt(e['pct_teeth_within_0p5mm_2deg'], '.1%')}",
        f"constraints  immobility violations {fmt(c['immobility_violation_rate'], '.1%')} of {c['immobile_teeth_checked']} held teeth",
        f"             worst penetration {fmt(c['worst_penetration_mm'])} mm | colliding pairs {c['colliding_adjacent_pairs']}",
        f"sensitivity  magnitude ratio {fmt(s['magnitude_ratio_mean'])} (1.0 = gold-sized deltas)",
        f"             direction cosine {fmt(s['direction_cosine_mean'])} (1.0 = right direction; ~0 = text-blind)",
    ]
    return "\n".join(lines)


def save_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=1))
