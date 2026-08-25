"""Generate task plans with the rich multimodal model trained by main_rich.py.

Examples:
    python starter/example_predict.py --split eval
    python starter/example_predict.py --split train --out submissions/predictions
    python starter/example_predict.py --split train --no-llm-stage1
    python starter/example_predict.py --split train --hard-llm-stage1-gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from contract import IDENTITY_Q, Plan, case_instructions, geodesic_deg, load_case, plan_filename, save_plan
from dataloader import load_sample
from dataloader import RichFeatureConfig, build_rich_features, collate_rich_samples
from multimodal_tooth_model import (
    ModelConfig,
    MultimodalToothMovementModel,
    numpy_batch_to_torch,
)

SNAP_TRANSLATION_MM = 0.05
SNAP_ROTATION_DEG = 1.0


def load_llm_fixed_teeth(case_dir: Path, instruction_id: str) -> set[int] | None:
    """Read movement_plan.json and return fixed_teeth for one instruction id."""
    plan_path = case_dir / "movement_plan.json"
    if not plan_path.exists():
        return None
    entries = json.loads(plan_path.read_text(encoding="utf-8"))
    for entry in entries:
        if entry.get("id") == instruction_id and "plan" in entry:
            return {int(fdi) for fdi in entry["plan"].get("fixed_teeth", [])}
    return None


def resolve_checkpoint_path(pack_dir: Path, checkpoint_arg: str | None) -> Path:
    """Resolve checkpoint path from explicit argument or common defaults."""
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        return path.resolve() if not path.is_absolute() else path

    candidates = [
        pack_dir / "starter" / "checkpoints_multimodal" / "best.pt",
        pack_dir / "checkpoints_multimodal" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a rich-model checkpoint. Tried: "
        + ", ".join(str(p) for p in candidates)
        + ". Pass --checkpoint explicitly."
    )


def predict_case_rich(model: MultimodalToothMovementModel,
                      case_dir: Path,
                      instruction: dict[str, Any],
                      device: torch.device,
                      feature_config: RichFeatureConfig,
                      movement_threshold: float,
                      hard_llm_gate: bool) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Predict one case/instruction pair and return plan transforms + run metadata."""
    raw_sample = load_sample(case_dir, instruction, include_targets=False)
    rich_sample = build_rich_features(raw_sample, feature_config)

    llm_fixed_teeth = None
    if hard_llm_gate:
        llm_fixed_teeth = load_llm_fixed_teeth(case_dir, instruction["id"])

    batch_np = collate_rich_samples([rich_sample])
    batch = numpy_batch_to_torch(batch_np, device)

    with torch.no_grad():
        outputs = model(batch)

    valid_mask = batch["batch_tooth_mask"][0].detach().cpu().numpy() > 0.5
    fdis = batch["fdis"][0].detach().cpu().numpy()
    move_probability = outputs["move_probability"][0].detach().cpu().numpy()
    translation = outputs["translation"][0].detach().cpu().numpy()
    rotation = outputs["rotation"][0].detach().cpu().numpy()
    instruction_features = batch["instruction_features"][0].detach().cpu().numpy()

    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, fdi in enumerate(fdis):
        if not valid_mask[index]:
            continue

        parser_protected = instruction_features[index, 13] > 0.5 if instruction_features.shape[1] >= 14 else False
        parser_movable = instruction_features[index, 14] > 0.5 if instruction_features.shape[1] >= 15 else True
        moves = move_probability[index] >= movement_threshold

        if hard_llm_gate and llm_fixed_teeth is not None and int(fdi) in llm_fixed_teeth:
            moves = False

        t = translation[index]
        q = rotation[index]
        if parser_protected or not parser_movable or not moves:
            transforms[int(fdi)] = (np.zeros(3), IDENTITY_Q.copy())
        elif (float(np.linalg.norm(t)) < SNAP_TRANSLATION_MM
              and geodesic_deg(q, IDENTITY_Q) < SNAP_ROTATION_DEG):
            transforms[int(fdi)] = (np.zeros(3), IDENTITY_Q.copy())
        else:
            transforms[int(fdi)] = (t, q)

    meta = {
        "llm_stage1_features_enabled": feature_config.use_llm_stage1,
        "llm_stage1_hard_gate_enabled": hard_llm_gate,
        "llm_stage1_fixed_teeth": sorted(llm_fixed_teeth) if llm_fixed_teeth else None,
    }
    return transforms, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--out", default="submissions/predictions_rich")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to main_rich checkpoint. If omitted, common defaults are tried.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--movement-threshold", type=float, default=0.5,
                    help="Move probability threshold for enabling movement heads.")
    ap.add_argument("--use-llm-stage1", dest="use_llm_stage1", action="store_true", default=True,
                    help="Use LLM Stage-1 features in rich feature construction (default: on).")
    ap.add_argument("--no-llm-stage1", dest="use_llm_stage1", action="store_false",
                    help="Disable LLM Stage-1 features for ablation.")
    ap.add_argument("--hard-llm-stage1-gate", action="store_true",
                    help="Additionally force LLM fixed_teeth to identity at inference (off by default).")
    args = ap.parse_args()

    pack_dir = Path(args.pack).resolve()
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = pack_dir / output_dir
    checkpoint_path = resolve_checkpoint_path(pack_dir, args.checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not look like main_rich output: {checkpoint_path}"
        )

    tooth_feature_dim = int(checkpoint["tooth_feature_dim"])
    pair_feature_dim = int(checkpoint["pair_feature_dim"])
    model_cfg = ModelConfig(**checkpoint.get("model_config", {}))
    device = torch.device(args.device)

    model = MultimodalToothMovementModel(
        tooth_feature_dim=tooth_feature_dim,
        pair_feature_dim=pair_feature_dim,
        config=model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_config = RichFeatureConfig(use_llm_stage1=args.use_llm_stage1)

    n = 0
    for case_dir in sorted((pack_dir / args.split).iterdir()):
        if not case_dir.is_dir():
            continue
        case = load_case(case_dir)
        for instruction in case_instructions(case_dir):
            transforms, meta = predict_case_rich(
                model=model,
                case_dir=case_dir,
                instruction=instruction,
                device=device,
                feature_config=feature_config,
                movement_threshold=args.movement_threshold,
                hard_llm_gate=args.hard_llm_stage1_gate,
            )
            plan = Plan(
                case_id=case.case_id,
                instruction_id=instruction["id"],
                instruction=instruction["text"],
                transforms=transforms,
                meta={
                    "model": "rich-multimodal",
                    "checkpoint": str(checkpoint_path),
                    "movement_threshold": args.movement_threshold,
                    **meta,
                },
            )
            save_plan(plan, output_dir / plan_filename(case.case_id, instruction["id"]))
            n += 1

    print(
        f"wrote {n} plans -> {output_dir} "
        f"(llm_stage1_features={args.use_llm_stage1}, hard_llm_gate={args.hard_llm_stage1_gate})"
    )


if __name__ == "__main__":
    main()
