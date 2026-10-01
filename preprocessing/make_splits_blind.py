#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path

import numpy as np

MASK_KEYS_DEFAULT = ["mask", "gt", "label", "y", "target"]

def iter_npz_files(patch_root: Path):
    # Support nested directories below patch_root.
    return sorted(patch_root.rglob("*.npz"))

def extract_line_id_from_patch_name(stem: str):
    """
    Patch filenames usually contain a line identifier such as HH06_23 or
    MSM135_P1015. Match HH06 identifiers first, then MSM135 identifiers, and
    return None when no supported identifier is found.
    """
    import re
    m = re.search(r"(HH06_\d{2}[ab]?)", stem)
    if m:
        return m.group(1)
    m = re.search(r"(MSM135_P\d{4})", stem)
    if m:
        return m.group(1)
    return None

def load_mask(npz_path: Path, mask_keys):
    d = np.load(npz_path, allow_pickle=False)
    for k in mask_keys:
        if k in d:
            return d[k]
    return None

def is_pos_patch(mask: np.ndarray, min_pos_px: int = 8):
    # Masks may contain 0, 1, and an ignore value such as 255.
    if mask is None:
        return False
    m = mask
    valid = (m != 255)
    if valid.sum() == 0:
        return False
    pos = ((m == 1) & valid).sum()
    return int(pos) >= int(min_pos_px)

def write_list(p: Path, items):
    p.write_text("\n".join(items) + ("\n" if items else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-root", required=True, type=str)
    ap.add_argument("--outdir", required=True, type=str)

    ap.add_argument("--blind-lines", nargs="+", required=True,
                    help="e.g. HH06_22 HH06_23 HH06_26 MSM135_P1005 MSM135_P1022")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Fraction of the non-blind patch pool assigned to validation.")

    ap.add_argument("--min-pos-px", type=int, default=8)
    ap.add_argument("--mask-keys", nargs="+", default=MASK_KEYS_DEFAULT)

    args = ap.parse_args()

    patch_root = Path(args.patch_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    blind_lines = set(args.blind_lines)
    rng = np.random.default_rng(args.seed)

    all_files = iter_npz_files(patch_root)
    if not all_files:
        raise RuntimeError(f"No npz found under: {patch_root}")

    # Collect blind and development pools.
    blind_all, pool_all = [], []
    unknown = []

    # Cache positive/negative classification.
    blind_pos, blind_neg = [], []
    pool_pos, pool_neg = [], []

    for f in all_files:
        line_id = extract_line_id_from_patch_name(f.stem)
        if line_id is None:
            unknown.append(str(f))
            continue

        mask = load_mask(f, args.mask_keys)
        posflag = is_pos_patch(mask, min_pos_px=args.min_pos_px)

        if line_id in blind_lines:
            blind_all.append(str(f))
            (blind_pos if posflag else blind_neg).append(str(f))
        else:
            pool_all.append(str(f))
            (pool_pos if posflag else pool_neg).append(str(f))

    # Split the development pool into training and validation patches.
    pool_all = np.array(pool_all, dtype=object)
    rng.shuffle(pool_all)

    n_val = int(round(len(pool_all) * float(args.val_ratio)))
    val = pool_all[:n_val].tolist()
    train = pool_all[n_val:].tolist()

    # train_shuffled
    train_shuffled = train.copy()
    rng.shuffle(train_shuffled)

    # Re-evaluate positive flags for consistent train/validation summaries.
    def split_pos_neg(file_list):
        pos, neg = [], []
        for fp in file_list:
            m = load_mask(Path(fp), args.mask_keys)
            if is_pos_patch(m, min_pos_px=args.min_pos_px):
                pos.append(fp)
            else:
                neg.append(fp)
        return pos, neg

    train_pos, train_neg = split_pos_neg(train)
    val_pos, val_neg = split_pos_neg(val)

    # Write split lists.
    write_list(outdir / "train_shuffled.txt", train_shuffled)
    write_list(outdir / "train_pos.txt", train_pos)
    write_list(outdir / "train_neg.txt", train_neg)

    write_list(outdir / "val.txt", val)
    write_list(outdir / "val_pos.txt", val_pos)
    write_list(outdir / "val_neg.txt", val_neg)

    write_list(outdir / "pool_pos.txt", pool_pos)
    write_list(outdir / "pool_neg.txt", pool_neg)

    write_list(outdir / "blind_all.txt", blind_all)
    write_list(outdir / "blind_pos.txt", blind_pos)
    write_list(outdir / "blind_neg.txt", blind_neg)

    summary = {
        "patch_root": str(patch_root),
        "outdir": str(outdir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "min_pos_px": args.min_pos_px,
        "mask_keys": args.mask_keys,
        "blind_lines": sorted(list(blind_lines)),
        "counts": {
            "total_files": len(all_files),
            "pool_all": len(pool_all),
            "blind_all": len(blind_all),
            "train": len(train),
            "val": len(val),
            "train_pos": len(train_pos),
            "train_neg": len(train_neg),
            "val_pos": len(val_pos),
            "val_neg": len(val_neg),
            "pool_pos": len(pool_pos),
            "pool_neg": len(pool_neg),
            "blind_pos": len(blind_pos),
            "blind_neg": len(blind_neg),
            "unknown": len(unknown),
        },
        "unknown_examples": unknown[:20],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[OK] wrote splits to:", outdir)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
