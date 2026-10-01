# utils.py
import os
import json
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_str():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_csv_row(path: str, header: List[str], row: Dict[str, Any]):
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in header) + "\n")


# ------------------------- Masked losses -------------------------
class MaskedBCEDiceLoss(nn.Module):
    """
    Binary segmentation:
    - logits: (B,1,H,W)
    - target: (B,1,H,W) in {0,1} with ignore=255 in raw mask
    - ignore_mask: (B,1,H,W) bool, True where valid
    """
    def __init__(self, bce_weight=0.7, dice_weight=0.3, pos_weight: Optional[float] = None, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target, valid_mask):
        # logits/target/valid_mask: (B,1,H,W)
        bce_map = self.bce(logits, target.float())
        bce = (bce_map * valid_mask.float()).sum() / (valid_mask.float().sum() + self.eps)

        prob = torch.sigmoid(logits)
        prob = prob * valid_mask
        tgt = target.float() * valid_mask

        # dice
        inter = (prob * tgt).sum(dim=(0, 2, 3))
        union = prob.sum(dim=(0, 2, 3)) + tgt.sum(dim=(0, 2, 3))
        dice = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice

# ------------------------- Masked Focal + Dice -------------------------
class MaskedFocalDiceLoss(nn.Module):
    """
    Binary segmentation (masked):
    - logits: (B,1,H,W)
    - target: (B,1,H,W) in {0,1}
    - valid_mask: (B,1,H,W) bool, True where valid (mask != 255)
    Focal loss computed from BCEWithLogits in a numerically stable way.
    """
    def __init__(
        self,
        focal_weight=0.7,
        dice_weight=0.3,
        alpha=0.25,     # class balance (pos)
        gamma=2.0,      # focusing
        eps=1e-6
    ):
        super().__init__()
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = eps

    def forward(self, logits, target, valid_mask):
        # logits/target/valid_mask: (B,1,H,W)
        tgt = target.float()
        vm = valid_mask.float()

        # BCE per-pixel
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")

        # pt = exp(-bce)  (stable)
        pt = torch.exp(-bce)

        # alpha_t: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * tgt + (1.0 - self.alpha) * (1.0 - tgt)

        focal = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        focal = (focal * vm).sum() / (vm.sum() + self.eps)

        # Dice (same masking logic as you already have)
        prob = torch.sigmoid(logits) * valid_mask
        tgtm = tgt * valid_mask

        inter = (prob * tgtm).sum(dim=(0, 2, 3))
        union = prob.sum(dim=(0, 2, 3)) + tgtm.sum(dim=(0, 2, 3))
        dice = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)
        dice = dice.mean()

        return self.focal_weight * focal + self.dice_weight * dice


class MaskedFocalLoss(nn.Module):
    """Masked sigmoid focal loss for thin boundary maps."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
        # logits/target: (B,1,H,W), valid_mask: (B,1,H,W) bool
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce

        vm = valid_mask.float()
        loss = loss * vm
        denom = vm.sum().clamp_min(1.0)
        return loss.sum() / denom

class MaskedHuberLoss(nn.Module):
    """Masked SmoothL1 (Huber) for SDF/DTM regression."""
    def __init__(self, delta: float = 0.2):
        super().__init__()
        self.delta = float(delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
        # pred/target: (B,1,H,W)
        vm = valid_mask.float()
        loss = F.smooth_l1_loss(pred, target, reduction="none", beta=self.delta)
        loss = loss * vm
        denom = vm.sum().clamp_min(1.0)
        return loss.sum() / denom

# ------------------------- Deep Supervision Wrapper -------------------------
def _downsample_nearest_float(x: torch.Tensor, size_hw: tuple):
    """Nearest neighbor downsample for masks (keeps 0/1 exactly)."""
    return F.interpolate(x, size=size_hw, mode="nearest")

class DeepSupervisionWrapper(nn.Module):
    """
    Wrap a base loss (MaskedBCEDiceLoss / MaskedFocalDiceLoss) to support
    deep supervision outputs: list of logits at multiple scales.

    Expected outputs order: [coarse, ..., fine] (fine is full-res).
    ds_weights should match outputs length.

    target: (B,1,H,W) float {0,1}
    valid_mask: (B,1,H,W) bool
    """
    def __init__(self, base_loss: nn.Module, ds_weights=(0.125, 0.25, 0.5, 1.0)):
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = [float(w) for w in ds_weights]

    def forward(self, logits, target, valid_mask):
        if not isinstance(logits, (list, tuple)):
            return self.base_loss(logits, target, valid_mask)

        if len(logits) != len(self.ds_weights):
            raise ValueError(f"len(logits)={len(logits)} must equal len(ds_weights)={len(self.ds_weights)}")

        total = 0.0
        for out, w in zip(logits, self.ds_weights):
            h, w2 = out.shape[-2], out.shape[-1]
            tgt_s = _downsample_nearest_float(target.float(), (h, w2))
            # valid is bool -> float -> downsample -> bool
            vm_s = _downsample_nearest_float(valid_mask.float(), (h, w2)) > 0.5
            total = total + w * self.base_loss(out, tgt_s, vm_s)

        return total

# ------------------------- Metrics (masked) -------------------------
@torch.no_grad()
def masked_confusion(prob: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, thr: float = 0.5):
    """
    prob: (B,1,H,W) float [0,1]
    target: (B,1,H,W) {0,1}
    valid_mask: bool
    """
    pred = (prob >= thr).bool() & valid_mask
    tgt = target.bool() & valid_mask

    tp = (pred & tgt).sum().item()
    fp = (pred & (~tgt)).sum().item()
    fn = ((~pred) & tgt).sum().item()
    tn = ((~pred) & (~tgt) & valid_mask).sum().item()
    return tp, fp, fn, tn


@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, thr: float = 0.5):
    prob = torch.sigmoid(logits)
    tp, fp, fn, tn = masked_confusion(prob, target, valid_mask, thr=thr)

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    # For binary segmentation, positive-class IoU is the reported IoU.
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "pos_iou": iou
    }


def plot_curves(metrics_csv: str, out_loss_png: str, out_metric_png: str):
    """
    Plot loss/metrics curves from metrics.csv WITHOUT pandas dependency.
    metrics.csv is written by save_csv_row().
    """
    import csv

    if not os.path.exists(metrics_csv):
        return

    rows = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return

    def get_float_list(key):
        vals = []
        for r in rows:
            v = r.get(key, "")
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        return vals

    epochs = get_float_list("epoch")

    # ---- loss curve ----
    plt.figure(figsize=(8, 4))
    if "train_loss" in rows[0]:
        plt.plot(epochs, get_float_list("train_loss"), label="train_loss")
    if "val_loss" in rows[0]:
        plt.plot(epochs, get_float_list("val_loss"), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_loss_png))
    plt.savefig(out_loss_png, dpi=200)
    plt.close()

    # ---- metric curve ----
    plt.figure(figsize=(8, 4))
    for k in ["val_iou", "val_dice", "val_precision", "val_recall"]:
        if k in rows[0]:
            plt.plot(epochs, get_float_list(k), label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_metric_png))
    plt.savefig(out_metric_png, dpi=200)
    plt.close()


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, cfg: Dict[str, Any]):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
