"""Multimodal tooth-movement model for dataloader_rich.py.

Inputs
------
* points_local: per-tooth point clouds
* tooth_features: geometry + instruction + arch/spacing/contralateral features
* pair_features: same-arch and cross-arch relational features
* fdis/jaw_index: tooth identity

Outputs
-------
* move_logits: probability that each tooth moves
* translation: predicted translation vector
* rotation: predicted unit quaternion [x, y, z, w]

This is a research baseline, not a clinically validated treatment planner.
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

from dataloader import RichFeatureConfig, make_torch_loader


@dataclass
class ModelConfig:
    d_model: int = 128
    point_dim: int = 96
    tooth_id_dim: int = 16
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.20
    max_fdi: int = 99
    translation_scale_mm: float = 8.0


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    move_translation_tol_mm: float = 1e-3
    move_rotation_tol_deg: float = 1e-2
    w_move: float = 0.25
    w_translation: float = 1.0
    w_rotation: float = 0.25
    w_fixed: float = 0.25


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_batch_to_torch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Convert numeric NumPy arrays produced by collate_rich_samples to tensors."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            out[key] = tensor.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


class PointCloudEncoder(nn.Module):
    """Shared PointNet-style encoder applied independently to every tooth."""

    def __init__(self, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, output_dim), nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.LayerNorm(output_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B,T,N,3]
        x = self.point_mlp(points)
        pooled = torch.cat([x.max(dim=2).values, x.mean(dim=2)], dim=-1)
        return self.output(pooled)


class PairBiasedSelfAttention(nn.Module):
    """Multi-head self-attention with learned tooth-pair bias and edge messages."""

    def __init__(self, d_model: int, pair_dim: int, num_heads: int,
                 dropout: float) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_dim, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, num_heads),
        )
        self.pair_value = nn.Sequential(
            nn.Linear(pair_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pair: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]  # [B,H,T,Dh]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self.pair_bias(pair).permute(0, 3, 1, 2)
        key_valid = valid_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~key_valid, torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=-1))

        edge_v = self.pair_value(pair).view(
            b, t, t, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        values = v[:, :, None, :, :] + edge_v
        context = torch.einsum("bhij,bhijd->bhid", attention, values)
        context = context.transpose(1, 2).reshape(b, t, d)
        context = self.out(context)
        return context * valid_mask.unsqueeze(-1)


class RelationalBlock(nn.Module):
    def __init__(self, d_model: int, pair_dim: int, num_heads: int,
                 dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PairBiasedSelfAttention(
            d_model, pair_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pair: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pair, valid_mask)
        x = x + self.ffn(self.norm2(x))
        return x * valid_mask.unsqueeze(-1)


class MultimodalToothMovementModel(nn.Module):
    """Point-cloud + tabular + instruction + relational movement model."""

    def __init__(self, tooth_feature_dim: int, pair_feature_dim: int,
                 config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.point_encoder = PointCloudEncoder(config.point_dim, config.dropout)
        self.tooth_id_embedding = nn.Embedding(config.max_fdi + 2, config.tooth_id_dim,
                                               padding_idx=0)
        self.jaw_embedding = nn.Embedding(3, 8, padding_idx=0)
        fusion_dim = config.point_dim + tooth_feature_dim + config.tooth_id_dim + 8
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.d_model),
            nn.LayerNorm(config.d_model), nn.GELU(), nn.Dropout(config.dropout),
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_feature_dim, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2), nn.GELU(),
        )
        projected_pair_dim = config.d_model // 2
        self.blocks = nn.ModuleList([
            RelationalBlock(config.d_model, projected_pair_dim,
                            config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.move_head = nn.Sequential(
            nn.Linear(config.d_model, 128), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )
        self.translation_head = nn.Sequential(
            nn.Linear(config.d_model, 128), nn.GELU(),
            nn.Linear(128, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(config.d_model, 128), nn.GELU(),
            nn.Linear(128, 4),
        )
        # Initialize rotations near the identity quaternion [0,0,0,1].
        nn.init.zeros_(self.rotation_head[-1].weight)
        with torch.no_grad():
            self.rotation_head[-1].bias.copy_(torch.tensor([0., 0., 0., 1.]))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        points = batch["points_local"].float()
        tooth_features = batch["tooth_features"].float()
        pair_features = batch["pair_features"].float()
        valid = batch.get("batch_tooth_mask", batch["tooth_mask"]).float()

        point_embedding = self.point_encoder(points)
        # Shift FDI and jaw indices by one, preserving index zero for padding.
        fdi = batch["fdis"].long()
        safe_fdi = torch.where(valid.bool(), fdi.clamp(0, self.config.max_fdi) + 1,
                               torch.zeros_like(fdi))
        jaw = batch["jaw_index"].long()
        safe_jaw = torch.where(valid.bool(), jaw.clamp(0, 1) + 1,
                               torch.zeros_like(jaw))
        x = torch.cat([
            point_embedding,
            tooth_features,
            self.tooth_id_embedding(safe_fdi),
            self.jaw_embedding(safe_jaw),
        ], dim=-1)
        x = self.fusion(x) * valid.unsqueeze(-1)
        pair = self.pair_projection(pair_features)
        for block in self.blocks:
            x = block(x, pair, valid)
        x = self.final_norm(x) * valid.unsqueeze(-1)

        move_logits = self.move_head(x).squeeze(-1)
        translation = torch.tanh(self.translation_head(x)) * self.config.translation_scale_mm
        rotation = F.normalize(self.rotation_head(x), dim=-1, eps=1e-8)
        return {
            "move_logits": move_logits,
            "move_probability": torch.sigmoid(move_logits),
            "translation": translation,
            "rotation": rotation,
            "tooth_embedding": x,
        }


def quaternion_geodesic_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sign-invariant quaternion angular error in degrees."""
    pred = F.normalize(pred, dim=-1, eps=1e-8)
    target = F.normalize(target, dim=-1, eps=1e-8)
    dot = torch.sum(pred * target, dim=-1).abs().clamp(max=1.0 - 1e-7)
    return torch.rad2deg(2.0 * torch.acos(dot))


def movement_targets(translation: torch.Tensor, rotation: torch.Tensor,
                     trans_tol: float, rot_tol_deg: float) -> torch.Tensor:
    trans_mag = torch.linalg.vector_norm(translation, dim=-1)
    identity = torch.zeros_like(rotation)
    identity[..., 3] = 1.0
    rot_deg = quaternion_geodesic_deg(rotation, identity)
    return ((trans_mag > trans_tol) | (rot_deg > rot_tol_deg)).float()


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                 config: TrainConfig) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch.get("batch_tooth_mask", batch["tooth_mask"]).float()
    target_t = batch["target_translation"].float()
    target_q = F.normalize(batch["target_rotation"].float(), dim=-1, eps=1e-8)
    moving = movement_targets(target_t, target_q,
                              config.move_translation_tol_mm,
                              config.move_rotation_tol_deg)

    move_loss = masked_mean(
        F.binary_cross_entropy_with_logits(outputs["move_logits"], moving,
                                           reduction="none"), valid)
    trans_per_tooth = F.smooth_l1_loss(outputs["translation"], target_t,
                                       reduction="none").mean(dim=-1)
    rot_per_tooth = quaternion_geodesic_deg(outputs["rotation"], target_q) / 180.0

    # Train movement magnitude/orientation most strongly on truly moving teeth.
    moving_valid = valid * moving
    translation_loss = masked_mean(trans_per_tooth, moving_valid)
    rotation_loss = masked_mean(rot_per_tooth, moving_valid)

    # Keep fixed teeth close to zero translation and identity rotation.
    fixed_valid = valid * (1.0 - moving)
    identity = torch.zeros_like(outputs["rotation"])
    identity[..., 3] = 1.0
    fixed_penalty = masked_mean(
        torch.linalg.vector_norm(outputs["translation"], dim=-1) +
        quaternion_geodesic_deg(outputs["rotation"], identity) / 180.0,
        fixed_valid)

    total = (config.w_move * move_loss +
             config.w_translation * translation_loss +
             config.w_rotation * rotation_loss +
             config.w_fixed * fixed_penalty)
    metrics = {
        "loss": float(total.detach()),
        "move_bce": float(move_loss.detach()),
        "translation_smooth_l1": float(translation_loss.detach()),
        "rotation_geodesic_scaled": float(rotation_loss.detach()),
        "fixed_penalty": float(fixed_penalty.detach()),
    }
    return total, metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: Any, device: torch.device,
             config: TrainConfig) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for raw_batch in loader:
        batch = numpy_batch_to_torch(raw_batch, device)
        outputs = model(batch)
        _, losses = compute_loss(outputs, batch, config)
        valid = batch.get("batch_tooth_mask", batch["tooth_mask"]).float()
        translation_error = torch.linalg.vector_norm(
            outputs["translation"] - batch["target_translation"].float(), dim=-1)
        rotation_error = quaternion_geodesic_deg(
            outputs["rotation"], batch["target_rotation"].float())
        target_move = movement_targets(
            batch["target_translation"].float(), batch["target_rotation"].float(),
            config.move_translation_tol_mm, config.move_rotation_tol_deg)
        pred_move = (outputs["move_probability"] >= 0.5).float()
        losses.update({
            "translation_mae_mm": float(masked_mean(translation_error, valid)),
            "rotation_mae_deg": float(masked_mean(rotation_error, valid)),
            "move_accuracy": float(masked_mean((pred_move == target_move).float(), valid)),
        })
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def infer_dimensions(loader: Any) -> tuple[int, int]:
    sample_batch = next(iter(loader))
    return int(sample_batch["tooth_features"].shape[-1]), int(sample_batch["pair_features"].shape[-1])


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, model_config: ModelConfig,
                    train_config: TrainConfig) -> None:
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }, path)


def train(pack_dir: str | Path, output_dir: str | Path,
          model_config: ModelConfig = ModelConfig(),
          train_config: TrainConfig = TrainConfig()) -> Path:
    seed_everything(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = RichFeatureConfig()
    train_loader = make_torch_loader(
        pack_dir, split="train", batch_size=train_config.batch_size,
        shuffle=True, num_workers=train_config.num_workers,
        config=feature_config)
    tooth_dim, pair_dim = infer_dimensions(train_loader)
    model = MultimodalToothMovementModel(tooth_dim, pair_dim, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay)

    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_path = output_dir / "best_multimodal_model.pt"
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        epoch_totals: dict[str, float] = {}
        steps = 0
        for raw_batch in train_loader:
            batch = numpy_batch_to_torch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, metrics = compute_loss(outputs, batch, train_config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            for key, value in metrics.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + value
            steps += 1
        summary = {key: value / max(steps, 1) for key, value in epoch_totals.items()}
        summary["epoch"] = float(epoch)
        history.append(summary)
        print(json.dumps(summary))
        if summary["loss"] < best_loss:
            best_loss = summary["loss"]
            save_checkpoint(best_path, model, optimizer, epoch,
                            model_config, train_config)

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    return best_path


@torch.no_grad()
def predict(model: nn.Module, raw_batch: dict[str, Any],
            device: torch.device | None = None,
            movement_threshold: float = 0.5) -> dict[str, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    batch = numpy_batch_to_torch(raw_batch, device)
    output = model(batch)
    valid = batch.get("batch_tooth_mask", batch["tooth_mask"]).bool()
    move = output["move_probability"] >= movement_threshold
    translation = output["translation"] * move.unsqueeze(-1)
    identity = torch.zeros_like(output["rotation"])
    identity[..., 3] = 1.0
    rotation = torch.where(move.unsqueeze(-1), output["rotation"], identity)
    return {
        "fdis": batch["fdis"].detach().cpu().numpy(),
        "valid_mask": valid.detach().cpu().numpy(),
        "move_probability": output["move_probability"].detach().cpu().numpy(),
        "translation": translation.detach().cpu().numpy(),
        "rotation": rotation.detach().cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal tooth movement model")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--output", default="model_outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.lr)
    checkpoint = train(args.pack, args.output, train_config=config)
    print(f"Saved best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
