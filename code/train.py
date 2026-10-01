# train.py (Fixed v9+LineBalanced: Context Manager for NPZ, No Double Norm, Correct Resume, Line-Balanced Sampler)
import os
import re
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict, Counter

from utils import (
    set_seed, ensure_dir,
    MaskedBCEDiceLoss, MaskedFocalDiceLoss, MaskedFocalLoss, MaskedHuberLoss, DeepSupervisionWrapper,
    compute_metrics_from_logits,
    save_checkpoint, load_checkpoint,
    save_csv_row, append_jsonl, plot_curves, save_json
)
from dataset_loader import NPZSegDataset, collect_npz_files
from augmentation import PatchAugmenter
from model import AttentionResUNet


# -----------------------------
# Balanced batch sampler helpers
# -----------------------------
def _infer_mask_key(npz_obj):
    # try common keys used in NPZ
    for k in ["mask", "gt", "label", "y", "target"]:
        if k in npz_obj.files:
            return k
    return None


def build_pos_neg_indices(npz_files, cache_json=None):
    """
    Determine pos/neg by whether mask contains any '1'.
    Cache result to speed up next runs.
    Uses a context manager to close file handles immediately.
    """
    if cache_json and os.path.exists(cache_json):
        print(f"[Cache] Loading pos/neg indices from {cache_json}")
        with open(cache_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d["pos"], d["neg"]

    pos, neg = [], []
    print(f"[Scan] Scanning {len(npz_files)} files for pos/neg balance...")

    for i, fp in enumerate(npz_files):
        try:
            # Use a context manager to prevent file descriptor exhaustion.
            with np.load(fp, allow_pickle=False) as z:
                k = _infer_mask_key(z)
                if k is None:
                    raise KeyError(f"No mask key found in {fp}, available={z.files}")
                m = z[k]
                # count only class==1 as positive; ignore 255
                is_pos = (m == 1).any()

            if is_pos:
                pos.append(i)
            else:
                neg.append(i)
        except Exception as e:
            print(f"[WARN] scan failed: {fp} -> {e}")
            neg.append(i)

        # Print progress every 2000 files
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{len(npz_files)} files...")

    print(f"[Scan] Done. Pos: {len(pos)}, Neg: {len(neg)}")

    if cache_json:
        ensure_dir(os.path.dirname(cache_json))
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump({"pos": pos, "neg": neg}, f)

    return pos, neg


class BalancedBatchSampler(Sampler):
    """
    Yield batches with fixed number of positive samples per batch.
    Finite iteration version.
    """
    def __init__(
        self,
        pos_indices,
        neg_indices,
        batch_size,
        pos_per_batch=4,
        drop_last=True,
        replace=True,
        seed=42,
        epoch_size=None,
    ):
        assert batch_size > 0
        assert 0 < pos_per_batch <= batch_size

        self.pos = list(pos_indices)
        self.neg = list(neg_indices)

        self.bs = int(batch_size)
        self.kp = int(pos_per_batch)
        self.kn = self.bs - self.kp

        self.drop_last = bool(drop_last)
        self.replace = bool(replace)

        # Set the worker seed for reproducible sampling.
        self.seed = seed
        self.epoch = 0
        self.rng = np.random.default_rng(seed)

        if len(self.pos) == 0:
            raise ValueError("No positive samples found! Check mask key / data.")
        if len(self.neg) == 0:
            print("[WARN] No negative samples found; will sample positives to fill batch.")
            self.neg = self.pos.copy()

        if epoch_size is None:
            epoch_size = len(self.pos) + len(self.neg)
        self.epoch_size = int(epoch_size)

        if self.epoch_size <= 0:
            self.epoch_size = self.bs

    def set_epoch(self, epoch):
        self.epoch = epoch
        # Optional: Reseed rng per epoch if you want deterministic shuffle per epoch
        # self.rng = np.random.default_rng(self.seed + epoch)

    def __len__(self):
        if self.drop_last:
            n_batches = self.epoch_size // self.bs
        else:
            n_batches = int(np.ceil(self.epoch_size / float(self.bs)))
        return max(1, int(n_batches))

    def __iter__(self):
        n_batches = len(self)

        if self.replace:
            pos_arr = np.asarray(self.pos, dtype=np.int64)
            neg_arr = np.asarray(self.neg, dtype=np.int64)

            for _ in range(n_batches):
                p_batch = self.rng.choice(pos_arr, size=self.kp, replace=True).tolist()
                n_batch = self.rng.choice(neg_arr, size=self.kn, replace=True).tolist()
                batch = p_batch + n_batch
                self.rng.shuffle(batch)
                yield batch
            return

        # replace=False logic
        pos = self.pos.copy()
        neg = self.neg.copy()
        self.rng.shuffle(pos)
        self.rng.shuffle(neg)

        p_ptr, n_ptr = 0, 0
        for _ in range(n_batches):
            if p_ptr + self.kp > len(pos) or n_ptr + self.kn > len(neg):
                if self.drop_last:
                    break
                self.rng.shuffle(pos)
                self.rng.shuffle(neg)
                p_ptr, n_ptr = 0, 0

            p_batch = pos[p_ptr:p_ptr + self.kp]; p_ptr += self.kp
            n_batch = neg[n_ptr:n_ptr + self.kn]; n_ptr += self.kn

            batch = p_batch + n_batch
            self.rng.shuffle(batch)
            yield batch


# -----------------------------
# Line-balanced sampler helpers
# -----------------------------
def infer_line_id_from_path(fp: str) -> str:
    """
    Extract line token from filename.

    Examples:
      HH06-07_reproc_Jonas_...   -> HH06_07
      HH06-44b_reproc_Jonas_...  -> HH06_44b
      MSM135_P1014_Reproc_MH...  -> MSM135_P1014
    """
    bn = os.path.basename(fp)

    m = re.search(r"(HH06[-_]\d{2}[ab]?)", bn, flags=re.IGNORECASE)
    if m:
        tok = m.group(1).upper().replace("-", "_")
        return tok

    m = re.search(r"(MSM135_P\d{4})", bn, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return "UNKNOWN"


def group_indices_by_line(files):
    mp = defaultdict(list)
    for i, fp in enumerate(files):
        lid = infer_line_id_from_path(fp)
        mp[lid].append(i)
    return mp


def _load_mask_from_npz(npz_path: str, mask_keys):
    with np.load(npz_path, allow_pickle=False) as z:
        for k in mask_keys:
            if k in z.files:
                return z[k]
    return None


def scan_pos_flags(train_files, pos_thr=0.5, min_pos_px=8, mask_keys=None, cache_json=None):
    """
    Scan each npz and return boolean flags (pos if pos_px >= min_pos_px).
    Optional cache to avoid rescanning each run.
    """
    if mask_keys is None:
        mask_keys = ["mask", "gt", "label", "y", "target"]

    if cache_json and os.path.exists(cache_json):
        print(f"[Cache] Loading pos_flags from {cache_json}")
        with open(cache_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        flags = np.asarray(d["pos_flags"], dtype=np.bool_)
        if len(flags) != len(train_files):
            print("[WARN] pos_flags cache length mismatch. Rescanning...")
        else:
            return flags

    flags = []
    print(f"[Sampler] Scanning {len(train_files)} files for pos_flags...")
    for i, fp in enumerate(train_files):
        m = _load_mask_from_npz(fp, mask_keys)
        if m is None:
            flags.append(False)
            continue
        m = np.asarray(m)
        valid = (m != 255)
        pos = (m > pos_thr) & valid
        flags.append(int(pos.sum()) >= int(min_pos_px))

        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i+1} files...")

    flags = np.asarray(flags, dtype=np.bool_)

    if cache_json:
        ensure_dir(os.path.dirname(cache_json))
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump({"pos_flags": flags.astype(int).tolist()}, f)

    return flags


class LineBalancedPosNegSampler(Sampler):
    """
    Line-balanced sampler:
      1) uniformly pick a line
      2) pick a sample within that line
      3) enforce a target pos_frac by preferring pos/neg within the chosen line

    Returns INDICES (not batches). Use with DataLoader(..., sampler=..., shuffle=False).
    """
    def __init__(
        self,
        files,
        pos_flags,            # bool array of length len(files): True if sample is positive
        num_samples=None,
        pos_frac=0.3,
        cap_per_line=None,
        seed=42,
    ):
        self.files = files
        self.pos_flags = np.asarray(pos_flags, dtype=np.bool_)
        self.rng = np.random.default_rng(seed)

        line2idx_all = group_indices_by_line(files)

        self.line2pos = {}
        self.line2neg = {}
        for lid, idxs in line2idx_all.items():
            idxs = list(idxs)
            if cap_per_line is not None and len(idxs) > int(cap_per_line):
                # random subset for fairness (seeded)
                idxs = self.rng.choice(np.asarray(idxs), size=int(cap_per_line), replace=False).tolist()

            idxs = np.asarray(idxs, dtype=np.int64)
            pos = idxs[self.pos_flags[idxs]].tolist()
            neg = idxs[(~self.pos_flags[idxs])].tolist()

            self.line2pos[lid] = pos
            self.line2neg[lid] = neg

        self.lines = list(self.line2pos.keys())
        if len(self.lines) == 0:
            raise ValueError("No lines found for line-balanced sampling.")

        self.pos_frac = float(pos_frac)
        self.num_samples = int(num_samples) if num_samples is not None else len(files)

    def __len__(self):
        return int(self.num_samples)

    def __iter__(self):
        for _ in range(self.num_samples):
            lid = self.rng.choice(self.lines)

            want_pos = (self.rng.random() < self.pos_frac)
            pos_list = self.line2pos.get(lid, [])
            neg_list = self.line2neg.get(lid, [])

            if want_pos and len(pos_list) > 0:
                yield int(self.rng.choice(pos_list))
            elif (not want_pos) and len(neg_list) > 0:
                yield int(self.rng.choice(neg_list))
            else:
                # fallback if chosen class missing in this line
                pool = pos_list if len(pos_list) > 0 else neg_list
                if len(pool) == 0:
                    # extreme fallback: pick any line then any sample
                    lid2 = self.rng.choice(self.lines)
                    pool2 = (self.line2pos[lid2] + self.line2neg[lid2])
                    yield int(self.rng.choice(pool2))
                else:
                    yield int(self.rng.choice(pool))


# -----------------------------
# File list helpers
# -----------------------------
def read_list_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        items = [x.strip() for x in f.readlines() if x.strip()]
    return items


def build_files(train_root, val_root, train_list, val_list):
    if train_list and val_list:
        return read_list_file(train_list), read_list_file(val_list)
    train_files = collect_npz_files(train_root)
    val_files = collect_npz_files(val_root)
    return train_files, val_files


# -----------------------------
# Metrics / eval helpers
# -----------------------------
def _valid_pixel_count(valid: torch.Tensor) -> float:
    if valid.dim() == 4:
        v = valid[:, 0]
    else:
        v = valid
    return float(v.sum().item())


def _pos_pixel_count(tgt: torch.Tensor, valid: torch.Tensor, thr: float = 0.5) -> float:
    if tgt.dim() == 4:
        t = tgt[:, 0]
    else:
        t = tgt
    if valid.dim() == 4:
        v = valid[:, 0]
    else:
        v = valid
    pos = (t > thr) & (v > 0)
    return float(pos.sum().item())


def _pick_last(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return x[-1]
    return x


@torch.no_grad()
def _eval_loop(model, loader, device, loss_fn, thr: float, aux_heads: bool,
               w_seg: float,
               w_edge: float, w_sdf: float, edge_loss_fn, sdf_loss_fn,
               max_steps: int = 0, val_debug: bool = False, val_debug_every: int = 0,
               amp_enabled: bool = False):
    model.eval()
    loss_sum = 0.0
    valid_px_sum = 0.0
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for i, batch in enumerate(loader, start=1):
        if max_steps > 0 and i > max_steps:
            break

        x = batch["x"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        edge_tgt = batch.get("edge", None)
        sdf_tgt = batch.get("sdf", None)
        if edge_tgt is not None:
            edge_tgt = edge_tgt.to(device, non_blocking=True)
        if sdf_tgt is not None:
            sdf_tgt = sdf_tgt.to(device, non_blocking=True)

        sdf_valid = batch.get("sdf_valid", None)
        if sdf_valid is not None:
            sdf_valid = sdf_valid.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(x)
        else:
            out = model(x)

        if isinstance(out, dict):
            seg_logits = out["seg"]
            edge_logits = out.get("edge", None)
            sdf_pred = out.get("sdf", None)
        else:
            seg_logits, edge_logits, sdf_pred = out, None, None

        # Ensure aux heads use full-res tensors (last DS level if list/tuple)
        edge_logits_last = _pick_last(edge_logits)
        sdf_pred_last = _pick_last(sdf_pred)

        seg_loss = loss_fn(seg_logits, tgt, valid)
        loss = float(w_seg) * seg_loss

        if aux_heads and (edge_logits is not None) and (edge_tgt is not None):
            loss = loss + float(w_edge) * edge_loss_fn(edge_logits_last, edge_tgt, valid)

        if aux_heads and (sdf_pred is not None) and (sdf_tgt is not None):
            loss = loss + float(w_sdf) * sdf_loss_fn(
                sdf_pred_last, sdf_tgt, (sdf_valid if sdf_valid is not None else valid)
            )

        valid_px = _valid_pixel_count(valid)
        if valid_px > 0:
            loss_sum += float(loss.item()) * valid_px
            valid_px_sum += valid_px

        logits_last = _pick_last(seg_logits)
        m = compute_metrics_from_logits(logits_last, tgt, valid, thr=thr)
        for k in agg:
            agg[k] += int(m[k])

        if val_debug:
            if val_debug_every > 0:
                do_print = (i % int(val_debug_every) == 0)
            else:
                do_print = (np.random.rand() < 0.02)
            if do_print:
                pos_px = _pos_pixel_count(tgt, valid, thr=0.5)
                pos_ratio = pos_px / max(valid_px, 1.0)
                print(f"[DBG] batch={i:04d} valid_px={valid_px:.0f} pos_ratio={pos_ratio:.4f} loss={loss.item():.4f}")

    loss_weighted = loss_sum / max(valid_px_sum, 1.0)

    eps = 1e-9
    tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    metrics = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice": float(dice),
    }
    return float(loss_weighted), metrics


def build_balanced_sampler(train_files, pos_frac=0.5, pos_thr=0.5, min_pos_px=8, mask_keys=None):
    """
    Original weighted sampler (pos/neg only). Kept for backward compatibility.
    """
    if mask_keys is None:
        mask_keys = ["mask", "gt", "label", "y", "target"]

    is_pos = []
    print(f"[Sampler] Scanning {len(train_files)} files for WeightedRandomSampler...")
    for i, fp in enumerate(train_files):
        m = _load_mask_from_npz(fp, mask_keys)
        if m is None:
            is_pos.append(False)
            continue

        m = np.asarray(m)
        valid = (m != 255)
        pos = (m > pos_thr) & valid
        pos_px = int(pos.sum())
        is_pos.append(pos_px >= int(min_pos_px))

        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i+1} files...")

    is_pos = np.array(is_pos, dtype=np.bool_)
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())

    if n_pos == 0 or n_neg == 0:
        weights = np.ones(len(train_files), dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_files),
            replacement=True
        )
        stats = {"n_pos": n_pos, "n_neg": n_neg, "pos_frac_eff": float(n_pos / max(len(train_files), 1))}
        return sampler, stats

    pos_w = float(pos_frac) / n_pos
    neg_w = float(1.0 - pos_frac) / n_neg
    weights = np.where(is_pos, pos_w, neg_w).astype(np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_files),
        replacement=True
    )
    stats = {"n_pos": n_pos, "n_neg": n_neg, "pos_frac_target": float(pos_frac)}
    return sampler, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=str, required=True)
    ap.add_argument("--val-root", type=str, required=True)
    ap.add_argument("--train-list", type=str, default="")
    ap.add_argument("--val-list", type=str, default="")

    ap.add_argument("--input-keys", nargs="+", default=["raw", "env", "rms", "ifreq"])
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--norm", type=str, default="bn", choices=["bn", "gn", "none"])
    ap.add_argument("--no-att", action="store_true")
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)

    ap.add_argument("--dropout-p", type=float, default=0.0, help="Dropout probability for ResBlocks")

    # deep supervision
    ap.add_argument("--deep-supervision", action="store_true")
    ap.add_argument("--aux-heads", action="store_true")
    ap.add_argument("--w-seg", type=float, default=1.0)
    ap.add_argument("--w-edge", type=float, default=0.2)
    ap.add_argument("--w-sdf", type=float, default=0.1)
    ap.add_argument("--sdf-delta", type=float, default=0.2)
    ap.add_argument("--boundary-radius", type=int, default=1)
    ap.add_argument("--sdf-clip", type=float, default=20.0)
    ap.add_argument("--sdf-band", type=float, default=0.20,
                    help="Band (in normalized sdf units) around boundary to apply SDF loss: |sdf| < band")
    ap.add_argument("--sdf-on-empty", action="store_true",
                    help="If set, still apply SDF loss on patches without positives (NOT recommended)")
    ap.add_argument("--ds-weights", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])

    ap.add_argument("--pos-weight", type=float, default=6.0)
    ap.add_argument("--bce-w", type=float, default=0.7)
    ap.add_argument("--dice-w", type=float, default=0.3)

    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="")

    ap.add_argument("--loss-type", type=str, default="bce_dice",
                    choices=["bce_dice", "focal_dice"])

    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-w", type=float, default=0.7)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--use-scheduler", action="store_true")
    ap.add_argument("--debug-steps", type=int, default=0)

    # sampler
    ap.add_argument("--use-sampler", action="store_true")
    ap.add_argument("--sampler-mode", type=str, default="lineposneg",
                    choices=["wposneg", "lineposneg"],
                    help="wposneg: WeightedRandomSampler(pos/neg). lineposneg: line-balanced + pos/neg control.")
    ap.add_argument("--pos-frac", type=float, default=0.5)
    ap.add_argument("--pos-thr", type=float, default=0.5)
    ap.add_argument("--min-pos-px", type=int, default=8)
    ap.add_argument("--mask-keys", nargs="+", default=["mask", "gt", "label", "y", "target"])
    ap.add_argument("--cap-per-line", type=int, default=300, help="Max patches per line for line-balanced sampler")
    ap.add_argument("--posflags-cache", type=str, default="", help="Optional json cache for pos_flags")

    ap.add_argument("--val-debug", action="store_true")
    ap.add_argument("--val-debug-every", type=int, default=0)
    ap.add_argument("--train-eval-steps", type=int, default=0)

    ap.add_argument("--balanced-batch", action="store_true")
    ap.add_argument("--pos-per-batch", type=int, default=4)

    args = ap.parse_args()
    set_seed(args.seed)

    ensure_dir(args.outdir)
    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    curve_dir = os.path.join(args.outdir, "curves")
    log_dir = os.path.join(args.outdir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(curve_dir)
    ensure_dir(log_dir)

    cfg = vars(args)
    save_json(os.path.join(args.outdir, "config.json"), cfg)

    train_files, val_files = build_files(args.train_root, args.val_root, args.train_list, args.val_list)
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    print(f"Input keys: {args.input_keys} | in_channels={len(args.input_keys)}")

    # debug: show line distribution (train)
    line_ids_sample = [infer_line_id_from_path(f) for f in train_files[:2000]]
    print("[DBG] train line_id top:", Counter(line_ids_sample).most_common(10))

    # Disable augmentation normalization to prevent duplicate normalization.
    aug = PatchAugmenter(use_albu=True, do_norm=False)

    train_ds = NPZSegDataset(
        train_files,
        input_keys=args.input_keys,
        augment=aug,
        boundary_radius=args.boundary_radius,
        sdf_clip=args.sdf_clip,
        sdf_band=args.sdf_band,
        sdf_on_empty=bool(args.sdf_on_empty),
        return_aux=bool(args.aux_heads)
    )
    val_ds = NPZSegDataset(
        val_files,
        input_keys=args.input_keys,
        augment=None,
        boundary_radius=args.boundary_radius,
        sdf_clip=args.sdf_clip,
        sdf_band=args.sdf_band,
        sdf_on_empty=bool(args.sdf_on_empty),
        return_aux=bool(args.aux_heads)
    )

    if args.balanced_batch:
        cache_json = os.path.join(log_dir, "posneg_cache.json")
        pos_idx, neg_idx = build_pos_neg_indices(train_files, cache_json=cache_json)
        print(f"[BalancedBatch] pos={len(pos_idx)} neg={len(neg_idx)} "
              f"pos_per_batch={args.pos_per_batch} bs={args.batch}")

        batch_sampler = BalancedBatchSampler(
            pos_idx, neg_idx,
            batch_size=args.batch,
            pos_per_batch=args.pos_per_batch,
            drop_last=True,
            replace=True,
            seed=args.seed,
            epoch_size=len(train_files),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        sampler = None
        if getattr(args, "use_sampler", False):
            if args.sampler_mode == "wposneg":
                sampler, st = build_balanced_sampler(
                    train_files=train_files,
                    pos_frac=float(args.pos_frac),
                    pos_thr=float(args.pos_thr),
                    min_pos_px=int(args.min_pos_px),
                    mask_keys=list(args.mask_keys),
                )
                print(
                    f"[SAMPLER] mode=wposneg | n_pos={st['n_pos']} n_neg={st['n_neg']} "
                    f"pos_frac_target={st.get('pos_frac_target', None)}"
                )
            else:
                cache_path = args.posflags_cache.strip()
                if not cache_path:
                    cache_path = os.path.join(log_dir, "posflags_cache.json")
                pos_flags = scan_pos_flags(
                    train_files=train_files,
                    pos_thr=float(args.pos_thr),
                    min_pos_px=int(args.min_pos_px),
                    mask_keys=list(args.mask_keys),
                    cache_json=cache_path
                )
                sampler = LineBalancedPosNegSampler(
                    files=train_files,
                    pos_flags=pos_flags,
                    num_samples=len(train_files),
                    pos_frac=float(args.pos_frac),
                    cap_per_line=int(args.cap_per_line) if args.cap_per_line > 0 else None,
                    seed=args.seed
                )
                print(
                    f"[SAMPLER] mode=lineposneg | pos_frac={args.pos_frac} "
                    f"cap_per_line={args.cap_per_line} cache={cache_path}"
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=False if sampler is not None else True,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionResUNet(
        in_channels=len(args.input_keys),
        base=args.base,
        norm=args.norm,
        use_att=(not args.no_att),
        deep_supervision=bool(args.deep_supervision),
        aux_heads=bool(args.aux_heads),
        dropout_p=args.dropout_p
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # loss
    if args.loss_type == "bce_dice":
        base_loss = MaskedBCEDiceLoss(
            bce_weight=args.bce_w,
            dice_weight=args.dice_w,
            pos_weight=args.pos_weight
        )
    else:
        base_loss = MaskedFocalDiceLoss(
            focal_weight=args.focal_w,
            dice_weight=args.dice_w,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma
        )
    base_loss = base_loss.to(device)

    if args.deep_supervision:
        loss_fn = DeepSupervisionWrapper(base_loss, ds_weights=args.ds_weights).to(device)
    else:
        loss_fn = base_loss

    edge_loss_fn = MaskedFocalLoss(alpha=0.25, gamma=2.0).to(device)
    sdf_loss_fn = MaskedHuberLoss(delta=args.sdf_delta).to(device)

    # AMP (train only)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    # Load Checkpoint & Start Epoch
    start_epoch = 1
    best_iou = -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer=opt, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_iou = float(ckpt.get("best_metric", -1.0))
        print(f"Resumed from {args.resume} @ epoch {start_epoch-1} best_iou={best_iou:.4f}")

    # Initialize the scheduler after resolving the resume epoch.
    scheduler = None
    if args.use_scheduler:
        last_epoch = start_epoch - 2 if (args.resume and start_epoch > 1) else -1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=1e-6, last_epoch=last_epoch
        )
        print(f"Scheduler: CosineAnnealingLR enabled (last_epoch={last_epoch}).")

    metrics_csv = os.path.join(log_dir, "metrics.csv")
    metrics_jsonl = os.path.join(log_dir, "metrics.jsonl")
    history_json = os.path.join(log_dir, "history.json")

    header = [
        "epoch", "lr",
        "train_loss", "val_loss",
        "val_iou", "val_dice", "val_precision", "val_recall",
        "train_eval_loss", "train_eval_iou"
    ]
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        # BalancedBatchSampler is batch_sampler, not sampler. Skip epoch control.
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        model.train()
        tr_losses = []

        for step, batch in enumerate(train_loader, start=1):
            if args.debug_steps > 0 and step > args.debug_steps:
                break

            x = batch["x"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            edge_tgt = batch.get("edge", None)
            sdf_tgt = batch.get("sdf", None)
            if edge_tgt is not None:
                edge_tgt = edge_tgt.to(device, non_blocking=True)
            if sdf_tgt is not None:
                sdf_tgt = sdf_tgt.to(device, non_blocking=True)

            sdf_valid = batch.get("sdf_valid", None)
            if sdf_valid is not None:
                sdf_valid = sdf_valid.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # ---- forward (AMP if enabled) ----
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    out = model(x)
                    if isinstance(out, dict):
                        seg_logits = out["seg"]
                        edge_logits = out.get("edge", None)
                        sdf_pred = out.get("sdf", None)
                    else:
                        seg_logits, edge_logits, sdf_pred = out, None, None

                    edge_logits_last = _pick_last(edge_logits)
                    sdf_pred_last = _pick_last(sdf_pred)

                    seg_loss = loss_fn(seg_logits, tgt, valid)
                    loss = float(args.w_seg) * seg_loss

                    if args.aux_heads and (edge_logits is not None) and (edge_tgt is not None):
                        loss = loss + float(args.w_edge) * edge_loss_fn(edge_logits_last, edge_tgt, valid)

                    if args.aux_heads and (sdf_pred is not None) and (sdf_tgt is not None):
                        loss = loss + float(args.w_sdf) * sdf_loss_fn(
                            sdf_pred_last, sdf_tgt, (sdf_valid if sdf_valid is not None else valid)
                        )
            else:
                out = model(x)
                if isinstance(out, dict):
                    seg_logits = out["seg"]
                    edge_logits = out.get("edge", None)
                    sdf_pred = out.get("sdf", None)
                else:
                    seg_logits, edge_logits, sdf_pred = out, None, None

                edge_logits_last = _pick_last(edge_logits)
                sdf_pred_last = _pick_last(sdf_pred)

                seg_loss = loss_fn(seg_logits, tgt, valid)
                loss = float(args.w_seg) * seg_loss

                if args.aux_heads and (edge_logits is not None) and (edge_tgt is not None):
                    loss = loss + float(args.w_edge) * edge_loss_fn(edge_logits_last, edge_tgt, valid)

                if args.aux_heads and (sdf_pred is not None) and (sdf_tgt is not None):
                    loss = loss + float(args.w_sdf) * sdf_loss_fn(
                        sdf_pred_last, sdf_tgt, (sdf_valid if sdf_valid is not None else valid)
                    )

            # ---- backward ----
            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                opt.step()

            tr_losses.append(float(loss.item()))

        train_loss = float(np.mean(tr_losses)) if tr_losses else 0.0

        # --- train-eval (diagnostic), pixel-weighted by valid_px, AMP OFF ---
        train_eval_loss, train_eval_metrics = 0.0, {"iou": 0.0}
        if args.train_eval_steps and args.train_eval_steps > 0:
            train_eval_loss, train_eval_metrics = _eval_loop(
                model=model,
                loader=train_loader,
                device=device,
                loss_fn=loss_fn,
                thr=args.thr,
                aux_heads=bool(args.aux_heads),
                w_seg=float(args.w_seg),
                w_edge=args.w_edge,
                w_sdf=args.w_sdf,
                edge_loss_fn=edge_loss_fn,
                sdf_loss_fn=sdf_loss_fn,
                max_steps=int(args.train_eval_steps),
                val_debug=False,
                val_debug_every=0,
                amp_enabled=False
            )

        # --- val (pixel-weighted loss, AMP OFF) ---
        val_loss, val_metrics = _eval_loop(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            thr=args.thr,
            aux_heads=bool(args.aux_heads),
            w_seg=float(args.w_seg),
            w_edge=args.w_edge,
            w_sdf=args.w_sdf,
            edge_loss_fn=edge_loss_fn,
            sdf_loss_fn=sdf_loss_fn,
            max_steps=0,
            val_debug=bool(args.val_debug),
            val_debug_every=int(args.val_debug_every),
            amp_enabled=False
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = float(opt.param_groups[0]["lr"])

        row = {
            "epoch": int(epoch),
            "lr": lr_now,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_iou": float(val_metrics["iou"]),
            "val_dice": float(val_metrics["dice"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "train_eval_loss": float(train_eval_loss),
            "train_eval_iou": float(train_eval_metrics["iou"]),
        }

        save_csv_row(metrics_csv, header, row)
        append_jsonl(metrics_jsonl, row)
        history.append(row)
        save_json(history_json, history)

        save_checkpoint(os.path.join(ckpt_dir, "last_model.pt"), model, opt, epoch, best_iou, cfg)
        if float(val_metrics["iou"]) > best_iou:
            best_iou = float(val_metrics["iou"])
            save_checkpoint(os.path.join(ckpt_dir, "best_model.pt"), model, opt, epoch, best_iou, cfg)

        plot_curves(
            metrics_csv=metrics_csv,
            out_loss_png=os.path.join(curve_dir, "loss_curve.png"),
            out_metric_png=os.path.join(curve_dir, "metrics_curve.png")
        )

        print(
            f"[{epoch:03d}/{args.epochs}] lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_iou={val_metrics['iou']:.4f} dice={val_metrics['dice']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"train_eval_loss={train_eval_loss:.4f} train_eval_iou={train_eval_metrics['iou']:.4f} "
            f"best_iou={best_iou:.4f}"
        )


if __name__ == "__main__":
    main()
