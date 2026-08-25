"""Train and validate the rich multimodal orthodontic movement model.

Compatible with:
    dataloader.py
    multimodal_tooth_model.py

Recommended objective:
    weighted move BCE
    + Smooth L1 translation loss on moving teeth
    + quaternion geodesic rotation loss on moving teeth
    + immobility penalty on fixed/protected teeth

The patient-level split keeps all instructions from one case in one partition.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from dataloader import (
    RichFeatureConfig,
    RichTreatmentPlanDataset,
    collate_rich_samples,
)
from multimodal_tooth_model import (
    ModelConfig,
    MultimodalToothMovementModel,
    numpy_batch_to_torch,
    quaternion_geodesic_deg,
)


@dataclass
class LossConfig:
    translation_beta_mm: float = 0.25
    move_translation_tol_mm: float = 1e-3
    move_rotation_tol_deg: float = 1e-2
    move_weight: float = 0.25
    translation_weight: float = 1.0
    rotation_weight: float = 0.30
    immobility_weight: float = 0.20
    protected_weight: float = 0.20
    max_move_pos_weight: float = 20.0


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.20
    grad_clip: float = 1.0
    patience: int = 20
    num_workers: int = 0
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def gold_move_mask(target_translation: torch.Tensor,
                   target_rotation: torch.Tensor,
                   translation_tol_mm: float,
                   rotation_tol_deg: float) -> torch.Tensor:
    translation_magnitude = torch.linalg.vector_norm(target_translation, dim=-1)
    identity = torch.zeros_like(target_rotation)
    identity[..., 3] = 1.0
    rotation_angle = quaternion_geodesic_deg(target_rotation, identity)
    return ((translation_magnitude > translation_tol_mm) |
            (rotation_angle > rotation_tol_deg)).float()


def protected_mask_from_instruction(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Combine parser-protected and LLM-protected flags.

    instruction_features layout from dataloader.py:
      0:7 global, 7:13 goals,
      13 parser_protected, 14 parser_movable,
      15 llm_protected, 16 llm_movable.
    """
    features = batch["instruction_features"].float()
    if features.shape[-1] < 17:
        return torch.zeros(features.shape[:2], device=features.device,
                           dtype=features.dtype)
    return torch.maximum(features[..., 13], features[..., 15])


class RichMovementLoss(nn.Module):
    """Composite objective matching MultimodalToothMovementModel outputs."""

    def __init__(self, config: LossConfig,
                 move_pos_weight: float = 1.0) -> None:
        super().__init__()
        self.config = config
        weight = min(max(float(move_pos_weight), 1.0),
                     config.max_move_pos_weight)
        self.register_buffer("move_pos_weight", torch.tensor(weight))

    def forward(self, prediction: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.config
        valid = batch.get("batch_tooth_mask", batch["tooth_mask"]).float()
        target_t = batch["target_translation"].float()
        target_q = F.normalize(batch["target_rotation"].float(), dim=-1, eps=1e-8)
        moving = gold_move_mask(
            target_t, target_q,
            cfg.move_translation_tol_mm,
            cfg.move_rotation_tol_deg,
        )
        fixed = 1.0 - moving
        protected = protected_mask_from_instruction(batch)

        move_values = F.binary_cross_entropy_with_logits(
            prediction["move_logits"], moving,
            pos_weight=self.move_pos_weight,
            reduction="none",
        )
        move_loss = masked_mean(move_values, valid)

        translation_values = F.smooth_l1_loss(
            prediction["translation"], target_t,
            beta=cfg.translation_beta_mm,
            reduction="none",
        ).mean(dim=-1)
        # Main regression is learned from teeth with non-zero gold movement.
        translation_loss = masked_mean(translation_values, valid * moving)

        rotation_values = quaternion_geodesic_deg(
            prediction["rotation"], target_q) / 180.0
        rotation_loss = masked_mean(rotation_values, valid * moving)

        identity = torch.zeros_like(prediction["rotation"])
        identity[..., 3] = 1.0
        predicted_translation_magnitude = torch.linalg.vector_norm(
            prediction["translation"], dim=-1)
        predicted_rotation_from_identity = quaternion_geodesic_deg(
            prediction["rotation"], identity) / 180.0
        no_motion_values = (predicted_translation_magnitude +
                            predicted_rotation_from_identity)
        immobility_loss = masked_mean(no_motion_values, valid * fixed)
        protected_loss = masked_mean(no_motion_values, valid * protected)

        total = (
            cfg.move_weight * move_loss
            + cfg.translation_weight * translation_loss
            + cfg.rotation_weight * rotation_loss
            + cfg.immobility_weight * immobility_loss
            + cfg.protected_weight * protected_loss
        )
        components = {
            "loss": total,
            "move_bce": move_loss,
            "translation_smooth_l1": translation_loss,
            "rotation_geodesic_scaled": rotation_loss,
            "immobility": immobility_loss,
            "protected": protected_loss,
        }
        return total, components


def case_id_from_sample_entry(entry: Any) -> str:
    """Read the patient/case ID from the base dataset's sample record."""
    case_path = entry[0]
    return Path(case_path).name


def build_loaders(pack_dir: Path, training: TrainingConfig,
                  feature_config: RichFeatureConfig) -> tuple[DataLoader, DataLoader]:
    train_all = RichTreatmentPlanDataset(
        pack_dir=pack_dir,
        split="train",
        include_targets=True,
        augment=True,
        config=feature_config,
        seed=training.seed,
    )
    validation_all = RichTreatmentPlanDataset(
        pack_dir=pack_dir,
        split="train",
        include_targets=True,
        augment=False,
        config=feature_config,
        seed=training.seed,
    )

    case_to_indices: dict[str, list[int]] = {}
    for index, entry in enumerate(train_all.samples):
        case_to_indices.setdefault(case_id_from_sample_entry(entry), []).append(index)
    case_ids = sorted(case_to_indices)
    if len(case_ids) < 2:
        raise ValueError("At least two patient cases are required for validation")

    shuffled = case_ids.copy()
    random.Random(training.seed).shuffle(shuffled)
    number_validation = max(
        1, int(round(len(case_ids) * training.validation_fraction)))
    number_validation = min(number_validation, len(case_ids) - 1)
    validation_cases = set(shuffled[:number_validation])

    train_indices = [
        index for case_id, indices in case_to_indices.items()
        if case_id not in validation_cases for index in indices
    ]
    validation_indices = [
        index for case_id, indices in case_to_indices.items()
        if case_id in validation_cases for index in indices
    ]

    train_dataset = Subset(train_all, train_indices)
    validation_dataset = Subset(validation_all, validation_indices)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_rich_samples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_rich_samples,
    )
    print(
        f"patients: {len(case_ids)} total, "
        f"{len(case_ids) - number_validation} train, "
        f"{number_validation} validation"
    )
    print(
        f"samples: {len(train_dataset)} train, "
        f"{len(validation_dataset)} validation"
    )
    return train_loader, validation_loader


@torch.no_grad()
def estimate_move_pos_weight(loader: DataLoader, loss_config: LossConfig) -> float:
    """Estimate negatives/positives on the training partition for weighted BCE."""
    positive = 0.0
    negative = 0.0
    for raw_batch in loader:
        target_t = torch.as_tensor(raw_batch["target_translation"]).float()
        target_q = torch.as_tensor(raw_batch["target_rotation"]).float()
        valid = torch.as_tensor(raw_batch["batch_tooth_mask"]).float()
        moving = gold_move_mask(
            target_t, target_q,
            loss_config.move_translation_tol_mm,
            loss_config.move_rotation_tol_deg,
        )
        positive += float((moving * valid).sum())
        negative += float(((1.0 - moving) * valid).sum())
    if positive == 0:
        raise ValueError("No moving teeth were found using the configured tolerances")
    return min(max(negative / positive, 1.0),
               loss_config.max_move_pos_weight)


def infer_feature_dimensions(loader: DataLoader) -> tuple[int, int]:
    raw_batch = next(iter(loader))
    return (int(raw_batch["tooth_features"].shape[-1]),
            int(raw_batch["pair_features"].shape[-1]))


def scalar_components(components: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in components.items()}


def accumulate(total: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + value


def average(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: AdamW, loss_fn: RichMovementLoss,
                    device: torch.device, grad_clip: float) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for raw_batch in loader:
        batch = numpy_batch_to_torch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss, components = loss_fn(prediction, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        accumulate(totals, scalar_components(components))
        steps += 1
    return average(totals, steps)


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader,
                       loss_fn: RichMovementLoss,
                       device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    for raw_batch in loader:
        batch = numpy_batch_to_torch(raw_batch, device)
        prediction = model(batch)
        _, components = loss_fn(prediction, batch)
        valid = batch["batch_tooth_mask"].float()
        target_t = batch["target_translation"].float()
        target_q = batch["target_rotation"].float()
        moving = gold_move_mask(
            target_t, target_q,
            loss_fn.config.move_translation_tol_mm,
            loss_fn.config.move_rotation_tol_deg,
        )
        predicted_moving = (prediction["move_probability"] >= 0.5).float()
        translation_error = torch.linalg.vector_norm(
            prediction["translation"] - target_t, dim=-1)
        rotation_error = quaternion_geodesic_deg(
            prediction["rotation"], target_q)
        values = scalar_components(components)
        values.update({
            "translation_mae": float(masked_mean(translation_error, valid)),
            "moving_translation_mae": float(masked_mean(
                translation_error, valid * moving)),
            "rotation_mae_deg": float(masked_mean(rotation_error, valid)),
            "move_accuracy": float(masked_mean(
                (predicted_moving == moving).float(), valid)),
        })
        accumulate(totals, values)
        steps += 1
    return average(totals, steps)


def save_checkpoint(path: Path, epoch: int, model: nn.Module,
                    optimizer: AdamW, scheduler: ReduceLROnPlateau,
                    train_metrics: dict[str, float],
                    validation_metrics: dict[str, float],
                    model_config: ModelConfig,
                    training_config: TrainingConfig,
                    loss_config: LossConfig,
                    tooth_feature_dim: int,
                    pair_feature_dim: int,
                    move_pos_weight: float) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "tooth_feature_dim": tooth_feature_dim,
        "pair_feature_dim": pair_feature_dim,
        "move_pos_weight": move_pos_weight,
    }, path)


def train(pack_dir: Path, checkpoint_dir: Path,
          model_config: ModelConfig,
          training_config: TrainingConfig,
          loss_config: LossConfig,
          feature_config: RichFeatureConfig) -> list[dict[str, Any]]:
    set_seed(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader = build_loaders(
        pack_dir, training_config, feature_config)
    tooth_feature_dim, pair_feature_dim = infer_feature_dimensions(train_loader)
    move_pos_weight = estimate_move_pos_weight(train_loader, loss_config)

    model = MultimodalToothMovementModel(
        tooth_feature_dim=tooth_feature_dim,
        pair_feature_dim=pair_feature_dim,
        config=model_config,
    ).to(device)
    loss_fn = RichMovementLoss(loss_config, move_pos_weight).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    print(f"device: {device}")
    print(f"tooth feature dimension: {tooth_feature_dim}")
    print(f"pair feature dimension: {pair_feature_dim}")
    print(f"move BCE positive weight: {move_pos_weight:.4f}")
    print(f"loss configuration: {asdict(loss_config)}")

    history: list[dict[str, Any]] = []
    best_validation = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, training_config.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, training_config.grad_clip)
        validation_metrics = validate_one_epoch(
            model, validation_loader, loss_fn, device)
        scheduler.step(validation_metrics["loss"])

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record))

        save_checkpoint(
            checkpoint_dir / "last.pt", epoch, model, optimizer, scheduler,
            train_metrics, validation_metrics, model_config, training_config,
            loss_config, tooth_feature_dim, pair_feature_dim, move_pos_weight)

        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt", epoch, model, optimizer, scheduler,
                train_metrics, validation_metrics, model_config, training_config,
                loss_config, tooth_feature_dim, pair_feature_dim, move_pos_weight)
            print(f"saved best checkpoint: {checkpoint_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1

        (checkpoint_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        if epochs_without_improvement >= training_config.patience:
            print(f"early stopping at epoch {epoch}")
            break
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train rich multimodal orthodontic movement model")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints_multimodal"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--translation-scale-mm", type=float, default=8.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--point-dim", type=int, default=96)
    parser.add_argument("--tooth-id-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--no-llm-stage1", action="store_true",
                        help="Exclude LLM move/fixed features; parser and geometry remain active.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
    )
    model_config = ModelConfig(
        d_model=args.d_model,
        point_dim=args.point_dim,
        tooth_id_dim=args.tooth_id_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        translation_scale_mm=args.translation_scale_mm)
    loss_config = LossConfig()
    feature_config = RichFeatureConfig(use_llm_stage1=not args.no_llm_stage1)

    train(
        pack_dir=args.pack,
        checkpoint_dir=args.checkpoints,
        model_config=model_config,
        training_config=training_config,
        loss_config=loss_config,
        feature_config=feature_config,
    )


if __name__ == "__main__":
    main()
