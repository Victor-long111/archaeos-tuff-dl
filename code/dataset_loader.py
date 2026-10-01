# dataset_loader.py
import os
import glob
import numpy as np
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from label_processor import mask_to_target_and_valid, mask_to_boundary, mask_to_sdf, IGNORE_VAL

# 

def find_key_case_insensitive(keys: List[str], candidates: List[str]) -> Optional[str]:
    keys_l = {k.lower(): k for k in keys}

    # 1) exact match first
    for cand in candidates:
        c = cand.lower()
        if c in keys_l:
            return keys_l[c]

    # 2) then substring match
    for cand in candidates:
        c = cand.lower()
        for k in keys:
            if c in k.lower():
                return k
    return None


def zscore_valid_aware(
    x: np.ndarray,
    mask: np.ndarray,
    ignore_val: int,
    eps: float = 1e-6,
    min_valid: int = 32,
) -> np.ndarray:
    """
    x: (C,H,W) float32
    mask: (H,W) int16 with {0,1,255}
    Only compute mean/std on valid pixels (mask != ignore_val).
    If valid pixels too few, fallback to full patch zscore.
    """
    assert x.ndim == 3, f"x must be (C,H,W), got {x.shape}"
    assert mask.ndim == 2, f"mask must be (H,W), got {mask.shape}"

    valid2d = (mask != ignore_val)
    n_valid = int(valid2d.sum())

    C = x.shape[0]

    if n_valid >= min_valid:
        # v: (C, N)
        v = x[:, valid2d]
        mean = v.mean(axis=1)  # (C,)
        std = v.std(axis=1)    # (C,)

        # reshape to (C,1,1) for broadcasting
        mean = mean.reshape(C, 1, 1)
        std = std.reshape(C, 1, 1)

        x = (x - mean) / (std + eps)
    else:
        # fallback: full patch stats
        mean = x.mean(axis=(1, 2), keepdims=True)  # (C,1,1)
        std = x.std(axis=(1, 2), keepdims=True)    # (C,1,1)
        x = (x - mean) / (std + eps)

    # Prevent shape drift
    assert x.ndim == 3 and x.shape[0] == C, f"norm output must be (C,H,W), got {x.shape}"
    return x


class NPZSegDataset(Dataset):
    """
    Read NPZ patches. Each NPZ must contain:
      - mask: (H,W) with {0,1,255}
      - input channels: by input_keys list, each is (H,W)
    """
    def __init__(
        self,
        npz_files: List[str],
        input_keys: List[str],
        augment=None,
        ignore_val: int = IGNORE_VAL,
        boundary_radius: int = 1,
        sdf_clip: float = 20.0,
        return_aux: bool = True,
        sdf_band: float = 0.20,
        sdf_on_empty: bool = False,
        use_valid_aware_norm: bool = True,
        min_valid_norm: int = 32,
    ):
        self.files = npz_files
        self.input_keys = input_keys
        self.augment = augment
        self.ignore_val = int(ignore_val)
        self.boundary_radius = int(boundary_radius)
        self.sdf_clip = float(sdf_clip)
        self.return_aux = bool(return_aux)

        self.sdf_band = float(sdf_band)
        self.sdf_on_empty = bool(sdf_on_empty)

        self.use_valid_aware_norm = bool(use_valid_aware_norm)
        self.min_valid_norm = int(min_valid_norm)

    def __len__(self):
        return len(self.files)

    def _load_one(self, fp: str) -> Dict[str, Any]:
        """
        Modified to use Context Manager (with statement) to ensure file handles are closed immediately.
        Also disabled pickle for security and speed.
        """
        # Use a context manager so the file handle is closed immediately.
        with np.load(fp, allow_pickle=False) as d:
            keys = list(d.files)

            msk_k = find_key_case_insensitive(keys, ["mask", "msk", "label", "gt"])
            if msk_k is None:
                raise KeyError(f"{os.path.basename(fp)} missing mask key.")

            mask = d[msk_k]
            if mask.ndim == 3:
                mask = np.squeeze(mask)
            mask = mask.astype(np.int16)

            H, W = mask.shape
            chans = []
            for name in self.input_keys:
                k = find_key_case_insensitive(keys, [name])
                if k is None:
                    raise KeyError(f"{os.path.basename(fp)} missing input key '{name}'")
                xi = d[k]
                if xi.ndim == 3:
                    xi = np.squeeze(xi)
                if xi.shape != (H, W):
                    raise ValueError(f"{os.path.basename(fp)} key '{k}' shape {xi.shape} != mask {(H, W)}")
                chans.append(xi.astype(np.float32))

            x = np.stack(chans, axis=0)  # (C,H,W)
            
        # context manager ends here, 'd' is closed automatically. 
        # 'x' and 'mask' are numpy arrays copied into memory, so they are safe to return.
        return {"x": x, "mask": mask}

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        try:
            sample = self._load_one(fp)
        except Exception as e:
            print(f"[ERROR] Failed to load {fp}: {e}")
            # Fallback strategy: pick a random other file to avoid crashing training
            # (Optional but recommended for robustness)
            new_idx = np.random.randint(len(self.files))
            return self.__getitem__(new_idx)

        if self.augment is not None:
            sample = self.augment(sample)

        x = sample["x"]       # (C,H,W)
        mask = sample["mask"] # (H,W)

        # target & valid
        target, valid = mask_to_target_and_valid(mask, ignore_val=self.ignore_val)

        # auxiliary labels
        if self.return_aux:
            edge = mask_to_boundary(mask, ignore_val=self.ignore_val, radius=self.boundary_radius)
            sdf = mask_to_sdf(mask, ignore_val=self.ignore_val, clip=self.sdf_clip)
            # sdf_valid: restrict SDF supervision to informative regions to prevent collapse on all-negative patches
            # - If patch has no positive pixels: by default disable SDF loss (sdf_valid all False)
            # - Otherwise, only supervise within a narrow band around the boundary: |sdf| < sdf_band
            has_pos = bool((target[0] > 0.5).any())
            if (not has_pos) and (not self.sdf_on_empty):
                sdf_valid = np.zeros_like(valid, dtype=bool)  # (1,H,W)
            else:
                sdf_valid = (valid > 0.5) & (np.abs(sdf) < self.sdf_band)
        else:
            edge, sdf = None, None

        # normalization
        # If normalization is enabled here, do not repeat it in augmentation.py.
        if self.use_valid_aware_norm:
            x = zscore_valid_aware(
                x, mask,
                ignore_val=self.ignore_val,
                eps=1e-6,
                min_valid=self.min_valid_norm
            )
        else:
            eps = 1e-6
            x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + eps)

        # torch tensors
        x_t = torch.from_numpy(x).float()         # (C,H,W)
        tgt_t = torch.from_numpy(target).float()  # (1,H,W)
        valid_t = torch.from_numpy(valid).bool()  # (1,H,W)

        out = {"x": x_t, "target": tgt_t, "valid": valid_t, "path": fp}

        if self.return_aux:
            out["edge"] = torch.from_numpy(edge).float()
            out["sdf"] = torch.from_numpy(sdf).float()

            out["sdf_valid"] = torch.from_numpy(sdf_valid.astype(np.bool_))

        return out


def collect_npz_files(root: str) -> List[str]:
    if os.path.isfile(root) and root.lower().endswith(".npz"):
        return [root]
    if os.path.isdir(root):
        return sorted(glob.glob(os.path.join(root, "**/*.npz"), recursive=True))
    raise FileNotFoundError(root)
