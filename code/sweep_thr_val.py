# sweep_thr_val.py
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_loader import NPZSegDataset
from model import AttentionResUNet
from utils import load_checkpoint, compute_metrics_from_logits


def read_list(txt_path: str):
    return [x.strip() for x in Path(txt_path).read_text().splitlines() if x.strip()]


@torch.no_grad()
def _forward_logits(model, x):
    """
    Compatible with:
      - plain seg output: Tensor (B,1,H,W)
      - deep supervision: list/tuple of tensors -> take last
      - aux heads: dict with key "seg" (and maybe "edge","sdf")
    """
    out = model(x)

    if isinstance(out, dict):
        seg = out.get("seg", None)
        if seg is None:
            raise KeyError("Model output dict has no key 'seg'.")
    else:
        seg = out

    # deep supervision: seg can be list/tuple of logits
    if isinstance(seg, (list, tuple)):
        logits = seg[-1]
    else:
        logits = seg

    return logits


@torch.no_grad()
def eval_thresholds(model, dl, thresholds, device):
    """
    Return:
      results: list of dicts per thr
      best: dict (best by IoU)
    """
    # Cache logits/tgt/valid on CPU to avoid repeated forward passes
    all_logits, all_tgt, all_valid = [], [], []

    model.eval()
    for b in dl:
        x = b["x"].to(device, non_blocking=True)
        tgt = b["target"].to(device, non_blocking=True)
        valid = b["valid"].to(device, non_blocking=True)

        logits = _forward_logits(model, x)  # (B,1,H,W)
        all_logits.append(logits.detach().cpu())
        all_tgt.append(tgt.detach().cpu())
        all_valid.append(valid.detach().cpu())

    results = []
    best = None

    for thr in thresholds:
        agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

        for logits, tgt, valid in zip(all_logits, all_tgt, all_valid):
            m = compute_metrics_from_logits(logits, tgt, valid, thr=float(thr))
            for k in agg:
                agg[k] += int(m[k])

        tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
        eps = 1e-9
        iou = tp / (tp + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        row = {
            "thr": float(thr),
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        results.append(row)

        if (best is None) or (row["iou"] > best["iou"]):
            best = row

    return results, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-list", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out-json", default="", type=str)

    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--num-workers", default=8, type=int)

    ap.add_argument("--base", default=32, type=int)
    ap.add_argument("--norm", default="gn", type=str)
    ap.add_argument("--use-att", action="store_true")
    ap.add_argument("--input-keys", nargs="+", default=["raw", "env", "rms"])

    # threshold sweep range
    ap.add_argument("--thr-min", default=0.01, type=float)
    ap.add_argument("--thr-max", default=0.30, type=float)
    ap.add_argument("--thr-step", default=0.01, type=float)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset / loader
    files = read_list(args.val_list)
    ds = NPZSegDataset(files, input_keys=args.input_keys, augment=None, return_aux=False)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # model: MUST match training ckpt structure (deep_supervision + aux_heads)
    model = AttentionResUNet(
        in_channels=len(args.input_keys),
        base=args.base,
        norm=args.norm,
        use_att=bool(args.use_att),
        deep_supervision=True,
        aux_heads=True,
    ).to(device)

    # load ckpt (strict load should now succeed)
    ckpt = load_checkpoint(args.ckpt, model, optimizer=None, map_location=device)
    print(f"Loaded ckpt: {args.ckpt}")
    if isinstance(ckpt, dict) and "best_metric" in ckpt:
        print(f"ckpt.best_metric (saved): {ckpt['best_metric']}")

    # thresholds
    tmin, tmax, tstep = args.thr_min, args.thr_max, args.thr_step
    n = int(round((tmax - tmin) / tstep)) + 1
    thresholds_fine = [round(tmin + i * tstep, 6) for i in range(n)]

    # coarse sweep
    thresholds_coarse = [round(i / 20, 2) for i in range(1, 20)]  # 0.05..0.95

    print(
        f"Fine sweep: {thresholds_fine[0]}..{thresholds_fine[-1]} step={tstep} (N={len(thresholds_fine)})"
    )
    print(
        f"Coarse sweep: {thresholds_coarse[0]}..{thresholds_coarse[-1]} (N={len(thresholds_coarse)})"
    )

    # eval
    print("\n=== Fine sweep ===")
    fine_res, fine_best = eval_thresholds(model, dl, thresholds_fine, device)
    for r in fine_res:
        print(
            f"thr={r['thr']:.2f} iou={r['iou']:.4f} dice={r['dice']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f}"
        )

    print("\n=== Coarse sweep ===")
    coarse_res, coarse_best = eval_thresholds(model, dl, thresholds_coarse, device)
    for r in coarse_res:
        print(
            f"thr={r['thr']:.2f} iou={r['iou']:.4f} dice={r['dice']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f}"
        )

    # pick final best (by IoU)
    best = fine_best if fine_best["iou"] >= coarse_best["iou"] else coarse_best

    print("\n==============================")
    print("BEST by IoU:")
    print(best)
    print("==============================\n")

    # save
    out_json = args.out_json
    if not out_json:
        out_json = str(Path(args.ckpt).with_suffix("").as_posix() + "_thr_sweep_val.json")

    payload = {
        "val_list": args.val_list,
        "ckpt": args.ckpt,
        "input_keys": args.input_keys,
        "fine": {
            "thr_min": tmin,
            "thr_max": tmax,
            "thr_step": tstep,
            "results": fine_res,
            "best": fine_best,
        },
        "coarse": {"results": coarse_res, "best": coarse_best},
        "best": best,
    }
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
