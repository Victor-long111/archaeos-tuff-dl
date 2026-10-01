# augmentation.py
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import albumentations as A
except Exception:
    A = None


def _amp_jitter_hwc(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ignore_val: int = 255,
    gain_range: Tuple[float, float] = (0.85, 1.15),
    bias_std: float = 0.02,
    per_channel: bool = True,
    p: float = 0.5,
) -> np.ndarray:
    """
    Amplitude gain/bias jitter for seismic attributes.
    FIXED: Bias is only applied to valid pixels (mask != ignore_val).
             to prevent zero-padding from becoming non-zero artifacts.
    """
    if np.random.rand() > p:
        return img

    out = img.astype(np.float32, copy=True)
    H, W, C = out.shape

    # 1. Gain (Multiplicative) - OK to apply everywhere (0 * gain = 0)
    if per_channel:
        gains = np.random.uniform(gain_range[0], gain_range[1], size=(1, 1, C)).astype(np.float32)
        biases = (np.random.randn(1, 1, C).astype(np.float32) * bias_std)
    else:
        g = float(np.random.uniform(gain_range[0], gain_range[1]))
        b = float(np.random.randn() * bias_std)
        gains = np.full((1, 1, C), g, dtype=np.float32)
        biases = np.full((1, 1, C), b, dtype=np.float32)

    out = out * gains

    # 2. Bias (Additive) - CRITICAL: Only apply to valid region
    if mask is not None:
        # mask: (H,W) -> valid: (H,W,1)
        valid_region = (mask != ignore_val)[..., None].astype(np.float32)
        out = out + (biases * valid_region)
    else:
        # If no mask provided, assume full image is valid (risky for padding)
        out = out + biases

    return out


class PatchAugmenter:
    """
    Seismic-Safe Augmentation V3 (Final)
    
    Principles:
    1. NO vertical non-linear warping (Time/Phase preservation).
    2. NO bias artifacts on zero-padding.
    3. Stronger CoarseDropout for robustness.
    """

    def __init__(
        self,
        use_albu: bool = True,
        ignore_val: int = 255,
        do_norm: bool = False, # Usually False if Dataset does normalization
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        eps: float = 1e-6,

        # Geometry
        p_hflip: float = 0.5,
        p_affine: float = 0.5, # Linear affine only
        
        # Noise / Occlusion
        p_noise: float = 0.3,
        p_dropout: float = 0.25, # Increased slightly as per advice
        
        # Amplitude
        p_amp: float = 0.8,
        amp_gain_range: Tuple[float, float] = (0.8, 1.2),
        amp_bias_std: float = 0.02, # Small bias
    ):
        self.ignore_val = int(ignore_val)
        self.do_norm = bool(do_norm)
        self.clip_percentiles = clip_percentiles
        self.eps = float(eps)

        # Amp params
        self.p_amp = p_amp
        self.amp_gain_range = amp_gain_range
        self.amp_bias_std = amp_bias_std

        self.use_albu = bool(use_albu) and (A is not None)

        if self.use_albu:
            tfs = []
            
            # 1. Horizontal Flip (Trace mirroring is usually physically OK)
            if p_hflip > 0:
                tfs.append(A.HorizontalFlip(p=p_hflip))

            # 2. Linear Affine ONLY (Safe Geometry)
            # Strictly limit Y-axis (Time) changes
            if p_affine > 0:
                tfs.append(A.ShiftScaleRotate(
                    shift_limit_x=0.06, # Shift traces
                    shift_limit_y=0.00, # LOCK Time shift
                    scale_limit=0.1,    # Zoom in/out
                    rotate_limit=5,     # Slight rotation (dip change)
                    border_mode=0,      # Zero padding
                    value=0,
                    mask_value=self.ignore_val,
                    p=p_affine
                ))

            # 3. Noise (SNR simulation)
            if p_noise > 0:
                tfs.append(A.GaussNoise(var_limit=(0.001, 0.015), p=p_noise))

            # 4. CoarseDropout (Forcing context learning)
            # Increased p to 0.25, but kept holes reasonably sized to not kill thin layers
            if p_dropout > 0:
                tfs.append(A.CoarseDropout(
                    max_holes=8, min_holes=1,
                    max_height=32, max_width=32,
                    min_height=8, min_width=8,
                    fill_value=0,
                    mask_fill_value=self.ignore_val,
                    p=p_dropout
                ))

            self.tf = A.Compose(tfs)
        else:
            self.tf = None

    def _robust_zscore(self, x: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        # ... (Previous z-score code remains unchanged) ...
        # Keep your original robust z-score logic here
        C, H, W = x.shape
        out = x.astype(np.float32, copy=True)
        p_lo, p_hi = self.clip_percentiles
        for c in range(C):
            xc = out[c]
            if valid_mask is not None and valid_mask.any():
                vals = xc[valid_mask]
            else:
                vals = xc.reshape(-1)
            if vals.size < 16:
                mean = float(vals.mean()) if vals.size > 0 else 0.0
                std = float(vals.std()) if vals.size > 0 else 1.0
                out[c] = (xc - mean) / (std + self.eps)
                continue
            lo = np.percentile(vals, p_lo)
            hi = np.percentile(vals, p_hi)
            if hi <= lo:
                lo, hi = float(vals.min()), float(vals.max())
            xc_clip = np.clip(xc, lo, hi)
            if valid_mask is not None and valid_mask.any():
                vals2 = xc_clip[valid_mask]
            else:
                vals2 = xc_clip.reshape(-1)
            mean = float(vals2.mean())
            std = float(vals2.std())
            out[c] = (xc_clip - mean) / (std + self.eps)
        return out

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        x = sample["x"]
        m = sample["mask"]

        # 1. Geometry & Noise (Albumentations)
        if self.tf is not None:
            x_hwc = np.transpose(x, (1, 2, 0)).astype(np.float32)
            out = self.tf(image=x_hwc, mask=m)
            x_hwc = out["image"].astype(np.float32)
            m = out["mask"].astype(np.int16)

            # 2. Amplitude Jitter (Safe: only on valid pixels)
            x_hwc = _amp_jitter_hwc(
                x_hwc, 
                mask=m, 
                ignore_val=self.ignore_val,
                gain_range=self.amp_gain_range,
                bias_std=self.amp_bias_std,
                p=self.p_amp
            )
            x = np.transpose(x_hwc, (2, 0, 1)).astype(np.float32)

        # 3. Normalization
        if self.do_norm:
            valid = (m != self.ignore_val)
            x = self._robust_zscore(x, valid_mask=valid)

        sample["x"] = x
        sample["mask"] = m
        return sample
