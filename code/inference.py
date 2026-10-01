# inference.py (v11: v10 + TRACE-WISE BEST-CONTINUITY + MULTI-WINDOW TIME GATING)
import os
import re
import glob
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import ensure_dir, load_checkpoint
from model import AttentionResUNet
from dataset_loader import find_key_case_insensitive, zscore_valid_aware
from label_processor import IGNORE_VAL


# -----------------------------
# basic helpers
# -----------------------------
def _ensure_2d(a):
    if a is None:
        return None
    if a.ndim == 3:
        a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expect 2D array, got shape={a.shape}")
    return a


def _pick_last(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return x[-1]
    return x


# -----------------------------
# attribute compute (optional)
# -----------------------------
def compute_env(raw_hw: np.ndarray, log1p: bool = True):
    try:
        from scipy.signal import hilbert
    except Exception as e:
        raise RuntimeError("scipy is required to compute env (scipy.signal.hilbert).") from e
    analytic = hilbert(raw_hw, axis=0)
    env = np.abs(analytic).astype(np.float32)
    if log1p:
        env = np.log1p(env).astype(np.float32)
    return env


def compute_rms(raw_hw: np.ndarray, win: int = 21):
    try:
        from scipy.ndimage import uniform_filter1d
    except Exception as e:
        raise RuntimeError("scipy is required to compute rms (scipy.ndimage.uniform_filter1d).") from e
    x2 = (raw_hw.astype(np.float32) ** 2)
    m = uniform_filter1d(x2, size=int(win), axis=0, mode="nearest")
    rms = np.sqrt(np.maximum(m, 0.0)).astype(np.float32)
    return rms


def compute_ifreq(raw_hw: np.ndarray,
                  dt_ms: float = 1.0,
                  smooth: int = 5,
                  clip: float = 250.0,
                  gate: bool = True,
                  gate_pct: float = 10.0):
    try:
        from scipy.signal import hilbert
        from scipy.ndimage import uniform_filter1d
    except Exception as e:
        raise RuntimeError("scipy is required to compute ifreq (hilbert + uniform_filter1d).") from e

    raw = raw_hw.astype(np.float32)
    analytic = hilbert(raw, axis=0)
    phase = np.unwrap(np.angle(analytic), axis=0).astype(np.float32)

    dphi = np.diff(phase, axis=0, prepend=phase[0:1, :])
    dt = float(dt_ms) / 1000.0
    dt = dt if dt > 0 else 1e-3
    ifreq = (dphi / (2.0 * np.pi * dt)).astype(np.float32)

    if smooth and int(smooth) > 1:
        ifreq = uniform_filter1d(ifreq, size=int(smooth), axis=0, mode="nearest").astype(np.float32)

    if gate:
        env = np.abs(analytic).astype(np.float32)
        thr = np.percentile(env, float(gate_pct))
        ifreq = np.where(env >= thr, ifreq, 0.0).astype(np.float32)

    ifreq = np.abs(ifreq)
    if clip and float(clip) > 0:
        ifreq = np.clip(ifreq, 0.0, float(clip)).astype(np.float32)
    return ifreq


# -----------------------------
# load full-line arrays
# -----------------------------
def load_line_arrays(npz_path: str, input_keys,
                     compute_missing: bool = False,
                     env_log1p: bool = True,
                     rms_win: int = 21,
                     dt_ms: float = 1.0,
                     ifreq_smooth: int = 5,
                     ifreq_clip: float = 250.0,
                     ifreq_gate: bool = True,
                     ifreq_gate_pct: float = 10.0,
                     save_augmented_npz: str = None):
    with np.load(npz_path, allow_pickle=False) as d:
        keys = list(d.files)

        msk_k = find_key_case_insensitive(keys, ["mask", "msk", "label", "gt"])
        mask = None
        if msk_k is not None:
            mask = _ensure_2d(d[msk_k]).astype(np.int16)

        raw_k = find_key_case_insensitive(keys, ["raw"])
        raw = None
        if raw_k is not None:
            raw = _ensure_2d(d[raw_k]).astype(np.float32)

        chans = []
        H = W = None
        computed = {}

        for name in input_keys:
            kk = find_key_case_insensitive(keys, [name])
            if kk is not None:
                x = _ensure_2d(d[kk]).astype(np.float32)
            else:
                if (not compute_missing) or (raw is None):
                    raise KeyError(f"{os.path.basename(npz_path)} missing key '{name}'")

                n = name.lower()
                if n == "env":
                    x = compute_env(raw, log1p=env_log1p)
                elif n == "rms":
                    x = compute_rms(raw, win=rms_win)
                elif n == "ifreq":
                    x = compute_ifreq(raw, dt_ms=dt_ms, smooth=ifreq_smooth,
                                      clip=ifreq_clip, gate=ifreq_gate, gate_pct=ifreq_gate_pct)
                elif n == "raw":
                    x = raw
                else:
                    raise KeyError(f"{os.path.basename(npz_path)} missing key '{name}' (no auto-compute rule).")
                computed[name] = x

            if H is None:
                H, W = x.shape
            if x.shape != (H, W):
                raise ValueError(f"key '{name}' shape {x.shape} != {(H, W)}")
            chans.append(x)

        x = np.stack(chans, axis=0)  # (C,H,W)

    if save_augmented_npz is not None and compute_missing and raw is not None:
        out = {"raw": raw.astype(np.float32)}
        for k, v in computed.items():
            out[k] = v.astype(np.float32)
        if mask is not None:
            out["mask"] = mask.astype(np.int16)
        np.savez_compressed(save_augmented_npz, **out)

    return x, mask


# -----------------------------
# valid mask from raw (for normalization)
# -----------------------------
def build_valid_mask_from_raw(raw_hw: np.ndarray, pct: float = 10.0, ignore_val: int = IGNORE_VAL):
    env = np.abs(raw_hw).astype(np.float32)
    thr = np.percentile(env, float(pct))
    valid = (env >= thr)
    m = np.zeros_like(raw_hw, dtype=np.int16)
    m[~valid] = int(ignore_val)
    return m


# -----------------------------
# invalid/water/blank suppression
# -----------------------------
def _local_std2d(a_hw: np.ndarray, win: int = 31):
    try:
        from scipy.ndimage import uniform_filter
    except Exception:
        s = float(np.std(a_hw))
        return np.full_like(a_hw, s, dtype=np.float32)

    a = a_hw.astype(np.float32)
    win = int(win)
    win = max(3, win)
    if win % 2 == 0:
        win += 1

    m1 = uniform_filter(a, size=win, mode="nearest")
    m2 = uniform_filter(a * a, size=win, mode="nearest")
    var = np.maximum(m2 - m1 * m1, 0.0)
    return np.sqrt(var).astype(np.float32)


def build_invalid_mask_from_raw_rms(raw_hw: np.ndarray,
                                   rms_hw: np.ndarray = None,
                                   win: int = 31,
                                   std_eps: float = 1e-4):
    raw_std = _local_std2d(raw_hw, win=win)
    invalid = (raw_std < float(std_eps))

    if rms_hw is not None:
        rms_std = _local_std2d(rms_hw, win=win)
        invalid = invalid | (rms_std < float(std_eps))

    return invalid.astype(bool)


# -----------------------------
# output debug + seg selection
# -----------------------------
def _summ_tensor(t: torch.Tensor):
    tt = t.detach().float()
    return {
        "shape": list(tt.shape),
        "min": float(tt.min().cpu()),
        "max": float(tt.max().cpu()),
        "mean": float(tt.mean().cpu()),
        "std": float(tt.std().cpu()),
    }


def debug_model_output(out):
    def _print(prefix, obj):
        if torch.is_tensor(obj):
            s = _summ_tensor(obj)
            print(f"{prefix} tensor shape={s['shape']} min={s['min']:.3f} max={s['max']:.3f} "
                  f"mean={s['mean']:.3f} std={s['std']:.3f}")
        elif isinstance(obj, (list, tuple)):
            print(f"{prefix} {type(obj).__name__} len={len(obj)}")
            for i, x in enumerate(obj):
                _print(prefix + f"[{i}]", x)
        elif isinstance(obj, dict):
            print(f"{prefix} dict keys={list(obj.keys())}")
            for k, v in obj.items():
                _print(prefix + f".{k}", v)
        else:
            print(f"{prefix} type={type(obj)}")

    _print("[DEBUG_OUT]", out)


def _pick_seg_from_listtuple_largest_res(out_list):
    cands = []
    for i, v in enumerate(out_list):
        if torch.is_tensor(v) and v.ndim == 4 and v.shape[1] == 1:
            H, W = int(v.shape[-2]), int(v.shape[-1])
            cands.append((H * W, i, v))
    if not cands:
        for i, v in enumerate(out_list):
            if torch.is_tensor(v):
                return v, i
        raise TypeError(f"No tensor found in output list/tuple: {[type(x) for x in out_list]}")

    cands.sort(key=lambda x: x[0])
    _, best_i, best_t = cands[-1]
    return best_t, best_i


def extract_seg_logits(out, force_seg_index: int = None, debug: bool = False):
    if debug:
        debug_model_output(out)

    if isinstance(out, dict):
        for k in ["seg", "logits", "out", "main"]:
            if k in out and out[k] is not None:
                v = out[k]
                if isinstance(v, (list, tuple)):
                    if force_seg_index is not None:
                        vv = v[int(force_seg_index)]
                        if not torch.is_tensor(vv):
                            raise TypeError(f"--force-seg-index got non-tensor at out['{k}'][{force_seg_index}]")
                        return vv
                    vv, idx = _pick_seg_from_listtuple_largest_res(v)
                    if debug:
                        print(f"[DEBUG] picked seg from dict key='{k}' list/tuple idx={idx}")
                    return vv
                if torch.is_tensor(v):
                    return v

        for _, v in out.items():
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Model output dict has no usable tensor keys: {list(out.keys())}")

    if isinstance(out, (list, tuple)):
        if force_seg_index is not None:
            v = out[int(force_seg_index)]
            if not torch.is_tensor(v):
                raise TypeError(f"--force-seg-index got non-tensor: type={type(v)}")
            return v
        v, idx = _pick_seg_from_listtuple_largest_res(out)
        if debug:
            print(f"[DEBUG] picked seg from list/tuple idx={idx}")
        return v

    if torch.is_tensor(out):
        return out

    raise TypeError(f"Unsupported model output type: {type(out)}")


# -----------------------------
# sdf conversions
# -----------------------------
def sdf_to_heat_boundary(sdf: np.ndarray, tau: float = 0.15):
    tau = max(float(tau), 1e-6)
    return np.exp(-np.abs(sdf) / tau).astype(np.float32)


def sdf_to_inside_prob(sdf: np.ndarray, tau: float = 0.15):
    tau = max(float(tau), 1e-6)
    return (1.0 / (1.0 + np.exp(-sdf / tau))).astype(np.float32)


# -----------------------------
# postprocess: morph + remove small/short components
# -----------------------------
def postprocess_binary(bin_hw: np.ndarray,
                       min_area: int = 0,
                       min_trace_span: int = 0,
                       min_time_span: int = 0,
                       do_close: bool = False,
                       close_y: int = 3,
                       close_x: int = 11,
                       do_fillholes: bool = False):
    try:
        from scipy.ndimage import label, binary_closing, binary_fill_holes
    except Exception:
        return bin_hw.astype(np.uint8)

    bw = bin_hw.astype(bool)

    if do_close:
        sy = max(1, int(close_y))
        sx = max(1, int(close_x))
        structure = np.ones((sy, sx), dtype=bool)
        bw = binary_closing(bw, structure=structure)

    if do_fillholes:
        bw = binary_fill_holes(bw)

    if (min_area <= 0) and (min_trace_span <= 0) and (min_time_span <= 0):
        return bw.astype(np.uint8)

    lab, n = label(bw.astype(np.int8))
    if n <= 0:
        return bw.astype(np.uint8)

    out = np.zeros_like(bin_hw, dtype=np.uint8)
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.size == 0:
            continue
        area = int(ys.size)
        span_x = int(xs.max() - xs.min() + 1)
        span_y = int(ys.max() - ys.min() + 1)

        if (min_area > 0 and area < min_area):
            continue
        if (min_trace_span > 0 and span_x < min_trace_span):
            continue
        if (min_time_span > 0 and span_y < min_time_span):
            continue

        out[ys, xs] = 1
    return out


# -----------------------------
# NEW: trace-wise best-continuity time-window gating
# -----------------------------
def build_tracewise_best_time_windows(prob_hw: np.ndarray,
                                      thr_low: float = 0.15,
                                      close_y: int = 3,
                                      close_x: int = 7,
                                      min_trace_span: int = 80,
                                      max_thickness: int = 160,
                                      base_halfwin: int = 25,
                                      halfwin_scale: float = 0.6,
                                      smooth_win: int = 51,
                                      score_mode: str = "span_meanprob",
                                      score_q: float = 70.0):
    """
    Procedure:
      1) Threshold at ``thr_low`` and apply closing to improve lateral continuity.
      2) Label 2-D components and calculate a lateral-continuity score for each.
      3) Retain the highest-scoring component on each trace. A selected component
         may contain multiple vertically separated segments on one trace.
      4) Combine all selected pixels into a candidate mask.
      5) Relabel the candidate mask to obtain laterally continuous AT segments.
      6) Fit and smooth a centerline for each segment, then expand it by ``halfwin``.
      7) Combine the windows and suppress probabilities outside them.
    """
    try:
        from scipy.ndimage import label, binary_closing, uniform_filter1d
    except Exception as e:
        raise RuntimeError("scipy required for trace-wise TW gating (label/binary_closing/uniform_filter1d).") from e

    H, W = prob_hw.shape
    prob = prob_hw.astype(np.float32)

    # 1) candidates + closing
    cand = (prob >= float(thr_low))
    structure = np.ones((max(1, int(close_y)), max(1, int(close_x))), dtype=bool)
    cand = binary_closing(cand, structure=structure)

    lab, n = label(cand.astype(np.int8))
    if n <= 0:
        return []

    # 2) compute component scores
    comps = []
    for cid in range(1, n + 1):
        ys, xs = np.where(lab == cid)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        span = int(x1 - x0 + 1)
        if span <= 0:
            continue

        pvals = prob[ys, xs]
        mean_p = float(np.mean(pvals)) if pvals.size else 0.0
        q_p = float(np.percentile(pvals, float(score_q))) if pvals.size else 0.0
        area = int(ys.size)

        if score_mode == "span":
            score = float(span)
        elif score_mode == "span_qprob":
            score = float(span) * q_p
        else:  # "span_meanprob"
            score = float(span) * mean_p

        comps.append({
            "cid": int(cid),
            "x0": x0, "x1": x1,
            "trace_span": span,
            "area": area,
            "mean_p": mean_p,
            "q_p": q_p,
            "score": float(score),
        })

    if len(comps) == 0:
        return []

    # 3) per-trace winner component id
    best_id = np.zeros((W,), dtype=np.int32)
    best_score = np.full((W,), -1e18, dtype=np.float32)

    for c in comps:
        cid = c["cid"]
        x0, x1 = c["x0"], c["x1"]
        sc = float(c["score"])
        for x in range(x0, x1 + 1):
            if sc > float(best_score[x]):
                best_score[x] = sc
                best_id[x] = cid

    # 4) build chosen mask: only pixels whose component == winner of that trace
    winner_ids = best_id[None, :]  # (1,W)
    chosen = cand & (lab == winner_ids)

    # 5) derive segments from chosen (may split to multiple segments)
    lab2, n2 = label(chosen.astype(np.int8))
    if n2 <= 0:
        return []

    windows = []
    for sid in range(1, n2 + 1):
        ys, xs = np.where(lab2 == sid)
        if ys.size == 0:
            continue

        x0, x1 = int(xs.min()), int(xs.max())
        trace_span = int(x1 - x0 + 1)
        if trace_span < int(min_trace_span):
            continue

        L = trace_span
        y_center = np.full((L,), np.nan, dtype=np.float32)
        thick = np.zeros((L,), dtype=np.float32)

        for i, x in enumerate(range(x0, x1 + 1)):
            yy = ys[xs == x]
            if yy.size == 0:
                continue
            thick[i] = float(yy.max() - yy.min() + 1)
            w = prob[yy, x] + 1e-6
            y_center[i] = float(np.sum(yy * w) / np.sum(w))

        valid = np.isfinite(y_center)
        if valid.sum() < max(10, int(0.25 * L)):
            continue

        thick_med = float(np.median(thick[valid]))
        if thick_med > float(max_thickness):
            continue

        # gap fill by interpolation
        xi = np.arange(L)
        y_fill = y_center.copy()
        y_fill[~valid] = np.interp(xi[~valid], xi[valid], y_fill[valid]).astype(np.float32)

        sw = int(smooth_win)
        sw = max(5, sw)
        if sw % 2 == 0:
            sw += 1
        y_smooth = uniform_filter1d(y_fill, size=sw, mode="nearest").astype(np.float32)

        # expanded time window (slightly)
        halfwin = float(base_halfwin + halfwin_scale * (0.5 * thick_med))
        halfwin = max(6.0, halfwin)

        meta = dict(
            sid=int(sid),
            x0=int(x0), x1=int(x1),
            trace_span=int(trace_span),
            thick_med=float(thick_med),
            halfwin=float(halfwin),
            score_mode=str(score_mode),
            score_q=float(score_q),
        )
        windows.append((x0, x1, y_smooth, halfwin, meta))

    return windows


def gate_prob_by_windows(prob_hw: np.ndarray, windows):
    H, W = prob_hw.shape
    keep = np.zeros((H, W), dtype=np.uint8)

    for (x0, x1, y_smooth, halfwin, meta) in windows:
        for i, x in enumerate(range(x0, x1 + 1)):
            yc = float(y_smooth[i])
            y0 = int(max(0, np.floor(yc - halfwin)))
            y1 = int(min(H - 1, np.ceil(yc + halfwin)))
            keep[y0:y1 + 1, x] = 1

    gated = prob_hw.astype(np.float32) * keep.astype(np.float32)
    return gated, keep


# -----------------------------
# sliding window inference
# -----------------------------
def sliding_window_infer(
    x_chw: np.ndarray,
    model,
    device: torch.device,
    patch=256,
    stride=128,
    norm_mask_hw=None,
    ignore_val=IGNORE_VAL,
    no_norm: bool = False,
    min_valid_norm: int = 32,
    return_aux: bool = False,
    pred_sdf_clip: float = 1.0,
    amp: bool = False,
    force_seg_index: int = None,
    debug_output: bool = False,
):
    C, H, W = x_chw.shape

    seg_prob = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    edge_prob = None
    sdf_map = None
    if return_aux:
        edge_prob = np.zeros((H, W), dtype=np.float32)
        sdf_map = np.zeros((H, W), dtype=np.float32)

    wy = np.hanning(patch).astype(np.float32)
    wx = np.hanning(patch).astype(np.float32)
    win = np.outer(wy, wx)
    win = win / (win.max() + 1e-6)

    max_top = max(0, H - patch)
    max_left = max(0, W - patch)

    model.eval()

    use_amp = bool(amp) and (device.type == "cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()

    printed_debug = False

    with torch.no_grad():
        top_list = list(range(0, max_top + 1, stride)) if max_top > 0 else [0]
        left_list = list(range(0, max_left + 1, stride)) if max_left > 0 else [0]

        for top in top_list:
            for left in left_list:
                t0 = min(top, max_top)
                l0 = min(left, max_left)

                crop = x_chw[:, t0:t0 + patch, l0:l0 + patch]  # (C,h,w)

                if not no_norm:
                    if norm_mask_hw is not None:
                        m_crop = norm_mask_hw[t0:t0 + crop.shape[1], l0:l0 + crop.shape[2]]
                        crop = zscore_valid_aware(
                            crop.astype(np.float32),
                            m_crop.astype(np.int16),
                            ignore_val=int(ignore_val),
                            eps=1e-6,
                            min_valid=int(min_valid_norm),
                        )
                    else:
                        eps = 1e-6
                        mean = crop.mean(axis=(1, 2), keepdims=True)
                        std = crop.std(axis=(1, 2), keepdims=True)
                        crop = (crop - mean) / (std + eps)

                ph = patch - crop.shape[1]
                pw = patch - crop.shape[2]
                crop_t = torch.from_numpy(crop[None, ...]).to(device)  # (1,C,h,w)
                if ph > 0 or pw > 0:
                    crop_t = F.pad(crop_t, (0, max(0, pw), 0, max(0, ph)), mode="replicate")

                with autocast_ctx:
                    out = model(crop_t)

                debug_now = bool(debug_output) and (not printed_debug)
                logits = extract_seg_logits(out, force_seg_index=force_seg_index, debug=debug_now)
                printed_debug = printed_debug or debug_now

                p = torch.sigmoid(logits)[0, 0].float().detach().cpu().numpy().astype(np.float32)

                if ph > 0 or pw > 0:
                    p = p[:crop.shape[1], :crop.shape[2]]
                    win_use = win[:crop.shape[1], :crop.shape[2]]
                else:
                    win_use = win

                seg_prob[t0:t0 + crop.shape[1], l0:l0 + crop.shape[2]] += p * win_use
                wsum[t0:t0 + crop.shape[1], l0:l0 + crop.shape[2]] += win_use

                if return_aux and isinstance(out, dict):
                    if "edge" in out and out["edge"] is not None:
                        e_logits = _pick_last(out["edge"])
                        if torch.is_tensor(e_logits):
                            e = torch.sigmoid(e_logits)[0, 0].float().detach().cpu().numpy().astype(np.float32)
                            if ph > 0 or pw > 0:
                                e = e[:crop.shape[1], :crop.shape[2]]
                            edge_prob[t0:t0 + crop.shape[1], l0:l0 + crop.shape[2]] += e * win_use

                    if "sdf" in out and out["sdf"] is not None:
                        s = _pick_last(out["sdf"])
                        if torch.is_tensor(s):
                            s = s[0, 0].float().detach().cpu().numpy().astype(np.float32)
                            if pred_sdf_clip is not None and float(pred_sdf_clip) > 0:
                                s = np.clip(s, -float(pred_sdf_clip), float(pred_sdf_clip))
                            if ph > 0 or pw > 0:
                                s = s[:crop.shape[1], :crop.shape[2]]
                            sdf_map[t0:t0 + crop.shape[1], l0:l0 + crop.shape[2]] += s * win_use

    seg_prob = seg_prob / (wsum + 1e-6)

    if not return_aux:
        return seg_prob

    edge_prob = edge_prob / (wsum + 1e-6)
    sdf_map = sdf_map / (wsum + 1e-6)
    return {"seg_prob": seg_prob, "edge_prob": edge_prob, "sdf": sdf_map}


# -----------------------------
# plotting
# -----------------------------
def qc_plot(raw_like: np.ndarray,
            show_map: np.ndarray,
            mask=None,
            thr=0.5,
            out_png=None,
            title="",
            cmap_overlay="turbo",
            overlay_alpha=0.45,
            draw_contour=True,
            contour_mask=None,
            draw_gt=False):
    H, W = show_map.shape
    plt.figure(figsize=(14, 6))

    if raw_like is not None:
        v = np.std(raw_like) * 2.0 + 1e-6
        plt.imshow(raw_like, cmap="gray", aspect="auto", vmin=-v, vmax=v)
    else:
        plt.imshow(np.zeros((H, W), np.float32), cmap="gray", aspect="auto")

    plt.imshow(show_map, cmap=cmap_overlay, alpha=float(overlay_alpha), aspect="auto")

    if draw_contour:
        if contour_mask is None:
            pred = (show_map >= float(thr)).astype(np.float32)
        else:
            pred = (contour_mask > 0).astype(np.float32)
        plt.contour(pred, levels=[0.5], colors="lime", linewidths=1.2)

    if draw_gt and (mask is not None):
        valid = (mask != IGNORE_VAL)
        pos = (mask == 1) & valid
        plt.contour(pos.astype(np.float32), levels=[0.5], colors="yellow", linewidths=1.2)

    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Time")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        plt.close()
    else:
        plt.show()


# -----------------------------
# file resolving (FIXED: _ and - compatible)
# -----------------------------
def _token_to_fuzzy_regex(token: str):
    t = token.strip()
    t = re.escape(t)
    t = t.replace(r"\_", r"[-_]").replace(r"\-", r"[-_]")
    return re.compile(rf"(^|[^A-Za-z0-9]){t}([^A-Za-z0-9]|$)", flags=re.IGNORECASE)


def resolve_line_files(line_npz, line_dir, lines):
    files = []
    if line_npz is not None and len(line_npz) > 0:
        files.extend(line_npz)

    if line_dir is not None:
        all_npz = glob.glob(os.path.join(line_dir, "**", "*.npz"), recursive=True)
        if lines is None or len(lines) == 0:
            files.extend(sorted(all_npz))
        else:
            for token in lines:
                pat = _token_to_fuzzy_regex(token)
                hit = [p for p in all_npz if pat.search(os.path.basename(p))]
                if len(hit) == 0:
                    t_simple = token.replace("_", "-")
                    hit = [p for p in all_npz if t_simple.lower() in os.path.basename(p).lower()]
                if len(hit) == 0:
                    print(f"[WARN] No npz matched token='{token}' under {line_dir}")
                else:
                    files.extend(sorted(hit))

    seen = set()
    uniq = []
    for p in files:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


# -----------------------------
# checkpoint arch inference
# -----------------------------
def infer_arch_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    keys = list(sd.keys()) if isinstance(sd, dict) else []

    has_ds = any(k.startswith("ds1.") or k.startswith("ds2.") or k.startswith("ds3.") for k in keys)
    has_edge = any(k.startswith("edge_head.") for k in keys)
    has_sdf = any(k.startswith("sdf_head.") for k in keys)
    has_aux = bool(has_edge or has_sdf)
    return has_ds, has_aux


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--line-npz", type=str, nargs="*", default=None)
    ap.add_argument("--line-dir", type=str, default=None)
    ap.add_argument("--lines", type=str, nargs="*", default=None)

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--input-keys", nargs="+", default=["raw", "env", "rms"])
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.55)

    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--norm", type=str, default="gn", choices=["bn", "gn", "none"])
    ap.add_argument("--no-att", action="store_true")

    ap.add_argument("--deep-supervision", action="store_true")
    ap.add_argument("--aux-heads", action="store_true")
    ap.add_argument("--auto-arch", action="store_true")
    ap.add_argument("--strict-load", action="store_true")

    ap.add_argument("--compute-missing", action="store_true")
    ap.add_argument("--save-augmented", action="store_true")

    ap.add_argument("--env-log1p", action="store_true")
    ap.add_argument("--rms-win", type=int, default=21)
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--ifreq-smooth", type=int, default=5)
    ap.add_argument("--ifreq-clip", type=float, default=250.0)
    ap.add_argument("--ifreq-gate", action="store_true")
    ap.add_argument("--ifreq-gate-pct", type=float, default=10.0)

    # normalization control
    ap.add_argument("--norm-mode", type=str, default="auto", choices=["auto", "raw", "mask", "none"])
    ap.add_argument("--min-valid-norm", type=int, default=32)
    ap.add_argument("--raw-valid-pct", type=float, default=10.0)

    # output / debug control
    ap.add_argument("--save-aux", action="store_true")
    ap.add_argument("--vis-mode", type=str, default="seg",
                    choices=["seg", "heat_boundary", "sdf_inside", "heat_seg"])
    ap.add_argument("--sdf-tau", type=float, default=0.12)
    ap.add_argument("--pred-sdf-clip", type=float, default=1.0)
    ap.add_argument("--overlay-alpha", type=float, default=0.45)
    ap.add_argument("--post-min-area", type=int, default=800)
    ap.add_argument("--post-min-trace-span", type=int, default=120)
    ap.add_argument("--post-min-time-span", type=int, default=12)

    # morph PP
    ap.add_argument("--pp-close", action="store_true",
                    help="Morphological closing before CC filtering.")
    ap.add_argument("--pp-close-y", type=int, default=3)
    ap.add_argument("--pp-close-x", type=int, default=11)
    ap.add_argument("--pp-fillholes", action="store_true",
                    help="Fill holes after closing.")

    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--debug-output", action="store_true",
                    help="Print model output structure once per line (first patch).")
    ap.add_argument("--force-seg-index", type=int, default=None,
                    help="Force which index is seg logits if output is list/tuple. For debug only.")

    # invalid/water suppression
    ap.add_argument("--suppress-invalid", action="store_true",
                    help="Zero out predictions in invalid (near-constant) raw/rms regions.")
    ap.add_argument("--invalid-local-win", type=int, default=31)
    ap.add_argument("--invalid-std-eps", type=float, default=1e-4)

    # NEW: trace-wise time-window gating
    ap.add_argument("--tw-gate", action="store_true",
                    help="Trace-wise pick best-continuity component then build per-segment time windows and suppress prob outside windows.")
    ap.add_argument("--tw-thr-low", type=float, default=0.15)
    ap.add_argument("--tw-close-y", type=int, default=3)
    ap.add_argument("--tw-close-x", type=int, default=7)
    ap.add_argument("--tw-min-trace-span", type=int, default=80)
    ap.add_argument("--tw-max-thickness", type=int, default=160)
    ap.add_argument("--tw-base-halfwin", type=int, default=25)
    ap.add_argument("--tw-halfwin-scale", type=float, default=0.6)
    ap.add_argument("--tw-smooth-win", type=int, default=51)

    # scoring for "best continuity"
    ap.add_argument("--tw-score-mode", type=str, default="span_meanprob",
                    choices=["span", "span_meanprob", "span_qprob"],
                    help="How to score continuity for per-trace winner components.")
    ap.add_argument("--tw-score-q", type=float, default=70.0,
                    help="Quantile used when tw-score-mode=span_qprob.")

    args = ap.parse_args()

    ensure_dir(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_ckpt, aux_ckpt = infer_arch_from_ckpt(args.ckpt)
    if args.auto_arch:
        deep_supervision = bool(args.deep_supervision or ds_ckpt)
        aux_heads = bool(args.aux_heads or aux_ckpt)
    else:
        deep_supervision = bool(args.deep_supervision)
        aux_heads = bool(args.aux_heads)

    if args.save_aux:
        aux_heads = True

    print(f"[ARCH] ckpt_has_ds={ds_ckpt} ckpt_has_aux={aux_ckpt} | "
          f"use_deep_supervision={deep_supervision} use_aux_heads={aux_heads}")

    model = AttentionResUNet(
        in_channels=len(args.input_keys),
        base=args.base,
        norm=args.norm,
        use_att=(not args.no_att),
        deep_supervision=deep_supervision,
        aux_heads=aux_heads,
    ).to(device)

    # load checkpoint
    try:
        load_checkpoint(args.ckpt, model, optimizer=None, map_location=device)
    except RuntimeError:
        if args.strict_load:
            raise
        print("[WARN] strict load failed, retry with strict=False via raw state_dict load...")
        ckpt = torch.load(args.ckpt, map_location=device)
        sd = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[WARN] strict=False load done.")
        if missing:
            print("[WARN] missing keys (first 20):", missing[:20])
        if unexpected:
            print("[WARN] unexpected keys (first 20):", unexpected[:20])

    print("Loaded:", args.ckpt)

    line_files = resolve_line_files(args.line_npz, args.line_dir, args.lines)
    if len(line_files) == 0:
        raise SystemExit("No input NPZ files found. Use --line-npz or --line-dir + --lines")
    print(f"Found {len(line_files)} line(s). Start inference...")

    for fp in line_files:
        bn = os.path.basename(fp).replace(".npz", "")
        subdir = os.path.join(args.outdir, bn)
        ensure_dir(subdir)

        aug_npz_path = os.path.join(subdir, f"{bn}_aug.npz") if args.save_augmented else None

        x, mask = load_line_arrays(
            fp, args.input_keys,
            compute_missing=bool(args.compute_missing),
            env_log1p=bool(args.env_log1p),
            rms_win=int(args.rms_win),
            dt_ms=float(args.dt_ms),
            ifreq_smooth=int(args.ifreq_smooth),
            ifreq_clip=float(args.ifreq_clip),
            ifreq_gate=bool(args.ifreq_gate),
            ifreq_gate_pct=float(args.ifreq_gate_pct),
            save_augmented_npz=aug_npz_path
        )

        # choose norm mask
        if args.norm_mode == "mask":
            norm_mask = mask
        elif args.norm_mode == "raw":
            norm_mask = build_valid_mask_from_raw(
                raw_hw=x[0], pct=float(args.raw_valid_pct), ignore_val=IGNORE_VAL
            )
        elif args.norm_mode == "auto":
            if mask is not None:
                norm_mask = mask
            else:
                norm_mask = build_valid_mask_from_raw(
                    raw_hw=x[0], pct=float(args.raw_valid_pct), ignore_val=IGNORE_VAL
                )
        else:
            norm_mask = None

        out = sliding_window_infer(
            x, model, device=device,
            patch=args.patch, stride=args.stride,
            norm_mask_hw=norm_mask,
            ignore_val=IGNORE_VAL,
            no_norm=(args.norm_mode == "none"),
            min_valid_norm=int(args.min_valid_norm),
            return_aux=bool(args.save_aux),
            pred_sdf_clip=float(args.pred_sdf_clip),
            amp=bool(args.amp),
            force_seg_index=args.force_seg_index,
            debug_output=bool(args.debug_output),
        )

        if isinstance(out, dict):
            seg_prob = out["seg_prob"]
            edge_prob = out.get("edge_prob", None)
            sdf_map = out.get("sdf", None)
        else:
            seg_prob = out
            edge_prob = None
            sdf_map = None

        heat_boundary = None
        inside_prob = None
        if sdf_map is not None:
            heat_boundary = sdf_to_heat_boundary(sdf_map, tau=float(args.sdf_tau))
            inside_prob = sdf_to_inside_prob(sdf_map, tau=float(args.sdf_tau))

        if args.vis_mode == "seg":
            show_map = seg_prob
        elif args.vis_mode == "heat_boundary":
            if heat_boundary is None:
                raise RuntimeError("vis-mode=heat_boundary requires sdf output. Use --save-aux.")
            show_map = heat_boundary
        elif args.vis_mode == "sdf_inside":
            if inside_prob is None:
                raise RuntimeError("vis-mode=sdf_inside requires sdf output. Use --save-aux.")
            show_map = inside_prob
        else:
            if heat_boundary is None:
                raise RuntimeError("vis-mode=heat_seg requires sdf output. Use --save-aux.")
            show_map = heat_boundary * seg_prob

        # --- suppress invalid (water/blank) BEFORE gating ---
        invalid_hw = None
        if args.suppress_invalid:
            raw_hw = x[0]
            rms_hw = None
            try:
                rms_idx = [k.lower() for k in args.input_keys].index("rms")
                rms_hw = x[rms_idx]
            except Exception:
                rms_hw = None

            invalid_hw = build_invalid_mask_from_raw_rms(
                raw_hw=raw_hw,
                rms_hw=rms_hw,
                win=int(args.invalid_local_win),
                std_eps=float(args.invalid_std_eps),
            )

            seg_prob[invalid_hw] = 0.0
            show_map[invalid_hw] = 0.0
            if edge_prob is not None:
                edge_prob[invalid_hw] = 0.0
            if sdf_map is not None:
                sdf_map[invalid_hw] = 0.0
            if heat_boundary is not None:
                heat_boundary[invalid_hw] = 0.0
            if inside_prob is not None:
                inside_prob[invalid_hw] = 0.0

            inv_frac = float(invalid_hw.mean())
            print(f"[INVALID_SUPPRESS] invalid_frac={inv_frac:.4f} win={args.invalid_local_win} eps={args.invalid_std_eps}")

        # --- NEW: trace-wise best-continuity time-window gating ---
        tw_keep = None
        if args.tw_gate:
            windows = build_tracewise_best_time_windows(
                seg_prob,
                thr_low=float(args.tw_thr_low),
                close_y=int(args.tw_close_y),
                close_x=int(args.tw_close_x),
                min_trace_span=int(args.tw_min_trace_span),
                max_thickness=int(args.tw_max_thickness),
                base_halfwin=int(args.tw_base_halfwin),
                halfwin_scale=float(args.tw_halfwin_scale),
                smooth_win=int(args.tw_smooth_win),
                score_mode=str(args.tw_score_mode),
                score_q=float(args.tw_score_q),
            )
            if len(windows) > 0:
                seg_prob, tw_keep = gate_prob_by_windows(seg_prob, windows)
                show_map = show_map * tw_keep.astype(np.float32)
                print(f"[TW_GATE_TRACEBEST] windows kept={len(windows)}")
                for w in windows[:5]:
                    print("  ", w[4])

                # gate aux maps as well (optional but consistent)
                if edge_prob is not None:
                    edge_prob = edge_prob * tw_keep.astype(np.float32)
                if sdf_map is not None:
                    sdf_map = sdf_map * tw_keep.astype(np.float32)
                if heat_boundary is not None:
                    heat_boundary = heat_boundary * tw_keep.astype(np.float32)
                if inside_prob is not None:
                    inside_prob = inside_prob * tw_keep.astype(np.float32)
            else:
                print("[TW_GATE_TRACEBEST] no qualified windows -> skip gating")

        # --- binary ---
        pred_bin = (show_map >= float(args.thr)).astype(np.uint8)

        # --- PP (morph + CC) ---
        pred_bin_pp = postprocess_binary(
            pred_bin,
            min_area=int(args.post_min_area),
            min_trace_span=int(args.post_min_trace_span),
            min_time_span=int(args.post_min_time_span),
            do_close=bool(args.pp_close),
            close_y=int(args.pp_close_y),
            close_x=int(args.pp_close_x),
            do_fillholes=bool(args.pp_fillholes),
        )

        # Save NPZ
        out_npz = os.path.join(subdir, f"{bn}_pred.npz")
        payload = {
            "seg_prob": seg_prob.astype(np.float32),
            "thr": float(args.thr),
            "input_keys": np.array(args.input_keys, dtype="U"),
            "vis_mode": np.array([str(args.vis_mode)], dtype="U"),
            "pred_bin": pred_bin.astype(np.uint8),
            "pred_bin_pp": pred_bin_pp.astype(np.uint8),
            "post_min_area": int(args.post_min_area),
            "post_min_trace_span": int(args.post_min_trace_span),
            "post_min_time_span": int(args.post_min_time_span),
            "pp_close": int(bool(args.pp_close)),
            "pp_close_y": int(args.pp_close_y),
            "pp_close_x": int(args.pp_close_x),
            "pp_fillholes": int(bool(args.pp_fillholes)),
            "norm_mode": np.array([str(args.norm_mode)], dtype="U"),
            "raw_valid_pct": float(args.raw_valid_pct),
            "force_seg_index": -1 if args.force_seg_index is None else int(args.force_seg_index),
            "suppress_invalid": int(bool(args.suppress_invalid)),
            "invalid_local_win": int(args.invalid_local_win),
            "invalid_std_eps": float(args.invalid_std_eps),

            "tw_gate": int(bool(args.tw_gate)),
            "tw_thr_low": float(args.tw_thr_low),
            "tw_close_y": int(args.tw_close_y),
            "tw_close_x": int(args.tw_close_x),
            "tw_min_trace_span": int(args.tw_min_trace_span),
            "tw_max_thickness": int(args.tw_max_thickness),
            "tw_base_halfwin": int(args.tw_base_halfwin),
            "tw_halfwin_scale": float(args.tw_halfwin_scale),
            "tw_smooth_win": int(args.tw_smooth_win),
            "tw_score_mode": np.array([str(args.tw_score_mode)], dtype="U"),
            "tw_score_q": float(args.tw_score_q),
        }
        if invalid_hw is not None:
            payload["invalid_mask"] = invalid_hw.astype(np.uint8)
        if tw_keep is not None:
            payload["tw_keep_mask"] = tw_keep.astype(np.uint8)

        if args.save_aux and edge_prob is not None:
            payload["edge_prob"] = edge_prob.astype(np.float32)
        if args.save_aux and sdf_map is not None:
            payload["sdf"] = sdf_map.astype(np.float32)
            payload["heat_boundary"] = heat_boundary.astype(np.float32)
            payload["inside_prob"] = inside_prob.astype(np.float32)
            payload["sdf_tau"] = float(args.sdf_tau)

        np.savez_compressed(out_npz, **payload)
        print("Saved:", out_npz)

        # ---- QC PNGs ----
        raw_like = x[0]

        out_png1 = os.path.join(subdir, f"{bn}_qc1_{args.vis_mode}_thr{args.thr}_raw+heat+bin.png")
        qc_plot(
            raw_like=raw_like,
            show_map=show_map.astype(np.float32),
            mask=mask,
            thr=float(args.thr),
            out_png=out_png1,
            title=f"{bn} | {args.vis_mode} | thr={args.thr} | norm={args.norm_mode} | contour=BIN",
            overlay_alpha=float(args.overlay_alpha),
            contour_mask=pred_bin,
            draw_gt=False,
        )
        print("QC1 png:", out_png1)

        out_png2 = os.path.join(subdir, f"{bn}_qc2_{args.vis_mode}_thr{args.thr}_raw+heat+pp.png")
        qc_plot(
            raw_like=raw_like,
            show_map=show_map.astype(np.float32),
            mask=mask,
            thr=float(args.thr),
            out_png=out_png2,
            title=f"{bn} | {args.vis_mode} | thr={args.thr} | norm={args.norm_mode} | "
                  f"PP(close={int(args.pp_close)} fill={int(args.pp_fillholes)} "
                  f"area>={args.post_min_area}, x>={args.post_min_trace_span}, y>={args.post_min_time_span})",
            overlay_alpha=float(args.overlay_alpha),
            contour_mask=pred_bin_pp,
            draw_gt=False,
        )
        print("QC2 png:", out_png2)

        out_png3 = os.path.join(subdir, f"{bn}_qc3_thr{args.thr}_raw+pp_fill.png")
        qc_plot(
            raw_like=raw_like,
            show_map=pred_bin_pp.astype(np.float32),
            mask=mask,
            thr=0.5,
            out_png=out_png3,
            title=f"{bn} | BIN_PP(fill) | thr={args.thr} | norm={args.norm_mode}",
            overlay_alpha=0.35,
            draw_contour=False,
            draw_gt=False,
        )
        print("QC3 png:", out_png3)


if __name__ == "__main__":
    main()
