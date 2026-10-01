# run_train.py (Full version: supports line-balanced sampler + w_seg + existing options)
import os
import argparse
import subprocess
from pathlib import Path
import yaml


def _as_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _norm_loss_type(loss_cfg: dict):
    """
    Map yaml loss.type to internal cli loss-type.
    Accept:
      - MaskedBCEDiceLoss / bce_dice / bce+dice
      - MaskedFocalDiceLoss / focal_dice / focal+dice
    """
    t = str(loss_cfg.get("type", "MaskedBCEDiceLoss")).strip().lower()
    t = t.replace("-", "_").replace("+", "_").replace(" ", "_")
    if "focal" in t:
        return "focal_dice"
    return "bce_dice"


def _get_any(d: dict, keys, default=None):
    """Return first existing key in d."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to yaml config")
    ap.add_argument("--dry-run", action="store_true", help="only print command, do not run")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp = cfg.get("experiment", {})
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    loss = cfg.get("loss", {})
    metrics = cfg.get("metrics", {})
    debug = cfg.get("debug", {})          # optional
    sampler_cfg = cfg.get("sampler", {})  # optional

    # ---- resolve train.py path (same folder as this run_train.py) ----
    this_dir = Path(__file__).resolve().parent
    train_py = this_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found at: {train_py}")

    # ---- required fields ----
    outdir = exp.get("outdir", "")
    if not outdir:
        raise ValueError("experiment.outdir is required in config")

    patch_root = data.get("patch_root", "")
    if not patch_root:
        raise ValueError("data.patch_root is required in config")

    # allow optional different val_root
    val_root = data.get("val_root", "") or patch_root

    train_list = data.get("train_list", "")
    val_list = data.get("val_list", "")

    input_keys = data.get("input_keys", ["raw", "env", "rms", "ifreq"])

    python_exe = os.environ.get("PYTHON", None) or "python"

    cmd = [
        python_exe, str(train_py),
        "--train-root", str(patch_root),
        "--val-root", str(val_root),
        "--outdir", str(outdir),
        "--epochs", str(train.get("epochs", 30)),
        "--batch", str(data.get("batch", 16)),
        "--lr", str(train.get("lr", 2e-4)),
        "--weight-decay", str(train.get("weight_decay", 1e-4)),
        "--num-workers", str(data.get("num_workers", 4)),
        "--thr", str(metrics.get("threshold", 0.5)),
        "--seed", str(exp.get("seed", 42)),
        "--base", str(model.get("base", 32)),
        "--norm", str(model.get("norm", "bn")),
        "--input-keys", *[str(k) for k in input_keys],
    ]

    # lists (preferred)
    if train_list and val_list:
        cmd += ["--train-list", str(train_list), "--val-list", str(val_list)]

    # attention
    use_att = _as_bool(model.get("use_attention", True), default=True)
    if not use_att:
        cmd.append("--no-att")

    # dropout
    dropout_p = model.get("dropout_p", None)
    if dropout_p is not None:
        cmd += ["--dropout-p", str(float(dropout_p))]

    # deep supervision
    ds_on = _as_bool(model.get("deep_supervision", False), default=False)
    if ds_on:
        cmd.append("--deep-supervision")
        ds_w = model.get("ds_weights", None) or loss.get("ds_weights", None) or [0.125, 0.25, 0.5, 1.0]
        cmd += ["--ds-weights", *[str(float(x)) for x in ds_w]]

    # auxiliary heads
    aux_on = _as_bool(model.get("aux_heads", False), default=False)
    if aux_on:
        cmd.append("--aux-heads")

        # w_seg (NEW): allow user to set seg weight even with aux heads
        w_seg = _get_any(loss, ["w_seg", "wSeg"], default=None)
        if w_seg is not None:
            cmd += ["--w-seg", str(float(w_seg))]

        w_edge = _get_any(loss, ["w_edge", "wEdge"], default=0.2)
        w_sdf = _get_any(loss, ["w_sdf", "wSdf"], default=0.1)
        sdf_delta = _get_any(loss, ["sdf_delta", "delta"], default=0.2)
        boundary_radius = _get_any(loss, ["boundary_radius", "radius"], default=1)
        sdf_clip = _get_any(loss, ["sdf_clip", "clip"], default=20.0)
        sdf_band = _get_any(loss, ["sdf_band", "band"], default=0.20)
        sdf_on_empty = _as_bool(_get_any(loss, ["sdf_on_empty"], default=False), default=False)

        cmd += ["--w-edge", str(float(w_edge))]
        cmd += ["--w-sdf", str(float(w_sdf))]
        cmd += ["--sdf-delta", str(float(sdf_delta))]
        cmd += ["--boundary-radius", str(int(boundary_radius))]
        cmd += ["--sdf-clip", str(float(sdf_clip))]
        cmd += ["--sdf-band", str(float(sdf_band))]
        if sdf_on_empty:
            cmd += ["--sdf-on-empty"]
    else:
        # even without aux heads, allow setting w_seg if user provides it
        w_seg = _get_any(loss, ["w_seg", "wSeg"], default=None)
        if w_seg is not None:
            cmd += ["--w-seg", str(float(w_seg))]

    # ---- loss branch ----
    loss_type = _norm_loss_type(loss)
    cmd += ["--loss-type", loss_type]

    if loss_type == "bce_dice":
        pos_w = _get_any(loss, ["pos_weight", "posWeight"], default=3.0)
        bce_w = _get_any(loss, ["bce_weight", "bce_w", "bceW"], default=0.7)
        dice_w = _get_any(loss, ["dice_weight", "dice_w", "diceW"], default=0.3)
        cmd += [
            "--pos-weight", str(float(pos_w)),
            "--bce-w", str(float(bce_w)),
            "--dice-w", str(float(dice_w)),
        ]
    else:
        focal_alpha = _get_any(loss, ["focal_alpha", "alpha"], default=0.25)
        focal_gamma = _get_any(loss, ["focal_gamma", "gamma"], default=2.0)
        focal_w = _get_any(loss, ["focal_weight", "focal_w", "focalW"], default=0.7)
        dice_w = _get_any(loss, ["dice_weight", "dice_w", "diceW"], default=0.3)
        cmd += [
            "--focal-alpha", str(float(focal_alpha)),
            "--focal-gamma", str(float(focal_gamma)),
            "--focal-w", str(float(focal_w)),
            "--dice-w", str(float(dice_w)),
        ]

    # grad clip
    if "grad_clip_norm" in train:
        cmd += ["--grad-clip-norm", str(float(train.get("grad_clip_norm")))]

    # amp / scheduler
    if _as_bool(train.get("amp", False), default=False):
        cmd.append("--amp")
    if _as_bool(train.get("use_scheduler", False), default=False):
        cmd.append("--use-scheduler")

    # resume
    resume = train.get("resume", "")
    if resume:
        cmd += ["--resume", str(resume)]

    # debug steps
    if "debug_steps" in debug:
        cmd += ["--debug-steps", str(int(debug.get("debug_steps", 0)))]

    # val debug
    if _as_bool(debug.get("val_debug", False), default=False):
        cmd.append("--val-debug")
        if "val_debug_every" in debug:
            cmd += ["--val-debug-every", str(int(debug.get("val_debug_every", 0)))]

    # train-eval loop
    if "train_eval_steps" in debug:
        cmd += ["--train-eval-steps", str(int(debug.get("train_eval_steps", 0)))]

    # ------------------------------------------------------------------
    # sampler switches
    # (A) batch-level balanced batches: BalancedBatchSampler
    # sampler:
    #   balanced_batch: true
    #   pos_per_batch: 8
    # (B) epoch-level sampling:
    #   - default: WeightedRandomSampler (pos/neg)
    #   - NEW: line-balanced sampler (sampler_mode=lineposneg), plus cap_per_line + posflags_cache
    # ------------------------------------------------------------------
    balanced_batch = _as_bool(sampler_cfg.get("balanced_batch", False), default=False)
    if balanced_batch:
        pos_per_batch = sampler_cfg.get("pos_per_batch", None)
        if pos_per_batch is None:
            bs = int(data.get("batch", 16))
            pos_per_batch = max(1, bs // 2)
        cmd += ["--balanced-batch", "--pos-per-batch", str(int(pos_per_batch))]
    else:
        use_sampler = _as_bool(sampler_cfg.get("use_sampler", False), default=False)
        if use_sampler:
            pos_frac = float(sampler_cfg.get("pos_frac", 0.5))
            pos_thr = float(sampler_cfg.get("pos_thr", 0.5))
            min_pos_px = int(sampler_cfg.get("min_pos_px", 8))
            mask_keys = sampler_cfg.get("mask_keys", ["mask", "gt", "label", "y", "target"])

            cmd += [
                "--use-sampler",
                "--pos-frac", str(pos_frac),
                "--pos-thr", str(pos_thr),
                "--min-pos-px", str(min_pos_px),
                "--mask-keys", *[str(k) for k in mask_keys],
            ]

            # Line-balanced sampler controls.
            sampler_mode = str(sampler_cfg.get("sampler_mode", "") or "").strip()
            cap_per_line = sampler_cfg.get("cap_per_line", None)
            posflags_cache = str(sampler_cfg.get("posflags_cache", "") or "").strip()

            if sampler_mode:
                cmd += ["--sampler-mode", sampler_mode]

            if cap_per_line is not None and str(cap_per_line).strip() != "":
                cmd += ["--cap-per-line", str(int(cap_per_line))]

            if posflags_cache:
                cmd += ["--posflags-cache", posflags_cache]

    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
