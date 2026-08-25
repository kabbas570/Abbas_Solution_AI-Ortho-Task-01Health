"""Score plans against gold, restricted to the seed=42 held-out validation cases only.

Replicates main.py's build_loaders() case split (same seed, same validation_fraction)
so model-comparison runs are scored on cases the checkpoint never trained on, instead of
the full train split (80% of which was memorized during training).

    python starter/score_holdout.py --plans submissions/predictions_with_llm
    python starter/score_holdout.py --plans submissions/predictions_no_llm
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from contract import case_instructions, load_case, load_gold, load_plan, plan_filename
from metrics import (
    TRANS_TOL_MM, ROT_TOL_DEG,
    _endpoint_errors, _immobile_violations, _penetration_proxy, _sensitivity, format_report,
)


def held_out_case_ids(pack_dir: Path, validation_fraction: float = 0.2, seed: int = 42) -> set[str]:
    """Reproduce main.py's build_loaders() validation-case split, without importing torch."""
    case_ids = sorted(
        case_dir.name for case_dir in (pack_dir / "train").iterdir()
        if case_dir.is_dir() and case_instructions(case_dir)
    )
    if len(case_ids) < 2:
        raise ValueError("At least two patient cases are required for validation")
    shuffled = case_ids.copy()
    random.Random(seed).shuffle(shuffled)
    count = max(1, int(round(len(case_ids) * validation_fraction)))
    count = min(count, len(case_ids) - 1)
    return set(shuffled[:count])


def score_subset(pack_dir: Path, plans_dir: Path, case_ids: set[str]) -> dict:
    """Same metrics as metrics.score_pack(), restricted to the given train case ids."""
    per_pair = []
    all_et, all_er = [], []
    imm_bad = imm_total = 0
    pen_worst, pen_pairs = 0.0, 0
    missing = []
    sens_per_case = []

    for case_dir in sorted((pack_dir / "train").iterdir()):
        if not case_dir.is_dir() or case_dir.name not in case_ids:
            continue
        case = load_case(case_dir)
        battery = case_instructions(case_dir)
        case_plans, case_golds = {}, {}
        for instr in battery:
            iid = instr["id"]
            gold_path = case_dir / "gold_transforms.json"
            if not gold_path.exists():
                continue
            gold = load_gold(gold_path)
            plan_path = plans_dir / plan_filename(case.case_id, iid)
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
    return {
        "split": f"train-holdout({len(case_ids)} cases)",
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--validation-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    case_ids = held_out_case_ids(args.pack, args.validation_fraction, args.seed)
    print(f"held-out validation cases (seed={args.seed}): {sorted(case_ids)}")
    report = score_subset(args.pack, args.plans, case_ids)
    print(format_report(report, name=str(args.plans)))


if __name__ == "__main__":
    main()
