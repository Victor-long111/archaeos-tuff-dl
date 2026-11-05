import os
import glob
import argparse
import numpy as np
import random
from tqdm import tqdm


class StratifiedNegAroundPosSampler:
    """
    NEG sampler (Time,Trace) with final strategy:

    0-250 ms: sparse random sampling in shallow/water intervals.
    250-2500 ms: uniform coverage outside the positive footprint.
    >2500 ms: sparse random sampling in deep intervals.

    Inputs:
      - line-level NPZ dir: contains mask + raw/env/rms/ifreq/... (Time,Trace)
      - positive patch dir: output of POS sampler (must have meta with orig_file/top/left)

    Outputs:
      - negative/water, negative/sediment, negative/deep
      - each patch saves ALL 2D channels from line-level NPZ + mask + meta + axis
    """

    def __init__(self, line_npz_dir, pos_patch_dir, output_root, seed=42):
        self.line_npz_dir = line_npz_dir
        self.pos_patch_dir = pos_patch_dir
        self.output_root = output_root

        random.seed(seed)
        np.random.seed(seed)

        # -----------------------------
        # Core
        # -----------------------------
        self.PATCH = 256
        self.IGNORE_VAL = 255
        self.MIN_VALID_RATIO = 0.5
        self.SAVE_META = True
        self.DEDUP = True
        self.SAVE_ALL_2D_CHANNELS = True

        # -----------------------------
        # Time windows (ms approximately equals sample index when dt_ms=1.0).
        # -----------------------------
        self.DT_MS = 1.0

        # Shallow interval: 0-250 ms.
        self.SHALLOW_MS0 = 0
        self.SHALLOW_MS1 = 250

        # Sediment interval: 250-2500 ms, outside the positive footprint.
        self.SED_MS0 = 250
        self.SED_MS1 = 2500

        # >2500ms deep
        self.DEEP_MS_START = 2500

        # -----------------------------
        # Sampling counts
        # -----------------------------
        self.N_SHALLOW_PER_LINE = 18   # Sparse random samples.
        self.N_DEEP_PER_LINE = 22      # Sparse random samples.
        self.RAND_JITTER = 32

        # Sediment uniform grid density (basic coverage)
        self.SED_STRIDE_T = 128
        self.SED_STRIDE_X = 128

        # Strictly forbid overlap with POS footprint (patch-level)
        self.SED_REQUIRE_NO_POS_OVERLAP = True
        self.MIN_ALLOWED_RATIO_IN_PATCH = 1.0

        # Optional buffer around POS footprint (dilate)
        self.POS_FOOTPRINT_BUFFER = 0

        # Whether shallow/deep also must be outside pos footprint
        self.AVOID_POS_FOOTPRINT_FOR_WATER_DEEP = True

        # -----------------------------
        # Output dirs
        # -----------------------------
        if os.path.basename(os.path.normpath(output_root)).lower() == "negative":
            self.dir_neg = output_root
        else:
            self.dir_neg = os.path.join(output_root, "negative")

        self.dir_wat = os.path.join(self.dir_neg, "water")
        self.dir_sed = os.path.join(self.dir_neg, "sediment")
        self.dir_dep = os.path.join(self.dir_neg, "deep")
        self._reset_dirs()

    def _reset_dirs(self):
        paths = [self.dir_wat, self.dir_sed, self.dir_dep]
        # Validate all destinations before creating any; never delete old outputs.
        for p in paths:
            if os.path.lexists(p) and (
                os.path.islink(p) or not os.path.isdir(p) or os.listdir(p)
            ):
                raise FileExistsError(f"Refusing to replace existing output: {p}. Use a new output directory.")
        for p in paths:
            os.makedirs(p, exist_ok=True)

    # -----------------------------
    # ms -> sample index
    # -----------------------------
    def _ms_to_samp(self, ms):
        return int(round(float(ms) / max(self.DT_MS, 1e-9)))

    # -----------------------------
    # Load arrays (all 2D channels + mask)
    # -----------------------------
    def _load_arrays(self, fp):
        try:
            d = np.load(fp, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            return None

        keys = list(d.files)

        def find_key(cands):
            for cand in cands:
                for k in keys:
                    if cand.lower() in k.lower():
                        return k
            return None

        msk_k = find_key(["mask", "msk", "label", "gt"])
        if msk_k is None:
            return None

        mask = d[msk_k]
        if isinstance(mask, np.ndarray) and mask.ndim == 3:
            mask = np.squeeze(mask)
        mask = mask.astype(np.int16)

        H, W = mask.shape
        arrs = {"mask": mask}

        if self.SAVE_ALL_2D_CHANNELS:
            for k in keys:
                if k == msk_k:
                    continue
                x = d[k]
                if isinstance(x, np.ndarray) and x.ndim == 3:
                    x = np.squeeze(x)
                if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape != (H, W):
                    continue

                if np.issubdtype(x.dtype, np.floating):
                    x = x.astype(np.float32)
                elif np.issubdtype(x.dtype, np.integer):
                    x = x.astype(np.int16)

                arrs[k] = x

        # sanity
        for name, x in arrs.items():
            if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape != (H, W):
                raise ValueError(f"{os.path.basename(fp)}: bad '{name}' shape={getattr(x,'shape',None)} != {(H,W)}")

        return arrs

    # -----------------------------
    # POS footprint from POS patches
    # -----------------------------
    def _build_pos_footprint_from_patches(self, line_stem: str, H: int, W: int):
        P = self.PATCH
        pos_mask = np.zeros((H, W), dtype=bool)

        files = glob.glob(os.path.join(self.pos_patch_dir, "*.npz"))
        if not files:
            return pos_mask, 0, 0

        used = 0
        skipped_other = 0

        for fp in files:
            try:
                d = np.load(fp, allow_pickle=True)
            except Exception:
                continue

            if "meta" not in d.files:
                continue
            meta_arr = d["meta"]
            if len(meta_arr) == 0 or not isinstance(meta_arr[0], dict):
                continue

            meta = meta_arr[0]
            orig = str(meta.get("orig_file", ""))

            if orig != line_stem:
                skipped_other += 1
                continue

            if "top" not in meta or "left" not in meta:
                continue

            top = int(meta["top"])
            left = int(meta["left"])
            top = max(0, min(H - P, top))
            left = max(0, min(W - P, left))

            pos_mask[top:top + P, left:left + P] = True
            used += 1

        return pos_mask, used, skipped_other

    # -----------------------------
    # Box dilation (no scipy)
    # -----------------------------
    @staticmethod
    def _dilate_box(b: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return b
        H, W = b.shape
        out = np.zeros_like(b, dtype=bool)
        bp = np.pad(b, ((radius, radius), (radius, radius)), mode="constant", constant_values=False)
        for dy in range(-radius, radius + 1):
            y0 = radius + dy
            ys = bp[y0:y0 + H, :]
            for dx in range(-radius, radius + 1):
                x0 = radius + dx
                out |= ys[:, x0:x0 + W]
        return out

    # -----------------------------
    # Helpers
    # -----------------------------
    def _valid_ratio(self, m):
        return float((m != self.IGNORE_VAL).mean())

    def _pos_pixels(self, m):
        valid = (m != self.IGNORE_VAL)
        return int(((m == 1) & valid).sum())

    def _clamp_top_left(self, top, left, H, W):
        P = self.PATCH
        top = max(0, min(H - P, int(top)))
        left = max(0, min(W - P, int(left)))
        return top, left

    def _save_patch(self, arrs: dict, save_path: str, meta: dict):
        P = self.PATCH
        top, left = meta["top"], meta["left"]
        out = {k: v[top:top + P, left:left + P] for k, v in arrs.items()}
        if self.SAVE_META:
            out["meta"] = np.array([meta], dtype=object)
            out["axis"] = np.bytes_(b"Time,Trace")
        np.savez_compressed(save_path, **out)

    # -----------------------------
    # Main per-line
    # -----------------------------
    def _process_one_line(self, fp):
        line_stem = os.path.basename(fp).replace(".npz", "")
        arrs = self._load_arrays(fp)
        if arrs is None:
            return 0, 0, 0, 0, 0

        mask = arrs["mask"]
        H, W = mask.shape
        P = self.PATCH
        if H < P or W < P:
            return 0, 0, 0, 0, 0

        valid = (mask != self.IGNORE_VAL)
        pos = (mask == 1) & valid

        # POS footprint (optional buffer)
        pos_footprint, pos_used, pos_skipped_other = self._build_pos_footprint_from_patches(line_stem, H, W)
        if self.POS_FOOTPRINT_BUFFER > 0:
            pos_footprint = self._dilate_box(pos_footprint, self.POS_FOOTPRINT_BUFFER)

        # time bounds (sample indices)
        shallow0 = self._ms_to_samp(self.SHALLOW_MS0)
        shallow1 = self._ms_to_samp(self.SHALLOW_MS1)
        sed0 = self._ms_to_samp(self.SED_MS0)
        sed1 = self._ms_to_samp(self.SED_MS1)
        deep0 = self._ms_to_samp(self.DEEP_MS_START)

        shallow0 = max(0, min(H - 1, shallow0))
        shallow1 = max(0, min(H - 1, shallow1))
        if shallow1 < shallow0:
            shallow0, shallow1 = shallow1, shallow0

        sed0 = max(0, min(H - 1, sed0))
        sed1 = max(0, min(H - 1, sed1))
        if sed1 < sed0:
            sed0, sed1 = sed1, sed0

        deep0 = max(0, min(H - 1, deep0))

        # Allowed masks
        shallow_allowed = np.zeros((H, W), dtype=bool)
        shallow_allowed[shallow0:shallow1 + 1, :] = True
        shallow_allowed &= valid
        shallow_allowed &= (~pos)
        if self.AVOID_POS_FOOTPRINT_FOR_WATER_DEEP:
            shallow_allowed &= (~pos_footprint)

        sediment_allowed = np.zeros((H, W), dtype=bool)
        sediment_allowed[sed0:sed1 + 1, :] = True
        sediment_allowed &= valid
        sediment_allowed &= (~pos)
        sediment_allowed &= ~pos_footprint

        deep_allowed = np.zeros((H, W), dtype=bool)
        deep_allowed[deep0:H, :] = True
        deep_allowed &= valid
        deep_allowed &= (~pos)
        if self.AVOID_POS_FOOTPRINT_FOR_WATER_DEEP:
            deep_allowed &= (~pos_footprint)

        used_coords = set()

        # -----------------------------
        # 1) Sediment: uniform grid coverage (250~2500) OUTSIDE POS footprint
        # -----------------------------
        saved_sed = 0
        top_start = sed0
        top_end = max(sed0, sed1 - P + 1)

        for top in range(top_start, top_end + 1, self.SED_STRIDE_T):
            for left in range(0, W - P + 1, self.SED_STRIDE_X):
                key = (top, left)
                if self.DEDUP and key in used_coords:
                    continue

                if self.SED_REQUIRE_NO_POS_OVERLAP:
                    if np.any(pos_footprint[top:top + P, left:left + P]):
                        continue

                p_msk = mask[top:top + P, left:left + P]
                if self._valid_ratio(p_msk) < self.MIN_VALID_RATIO:
                    continue
                if self._pos_pixels(p_msk) > 0:
                    continue

                p_allowed = sediment_allowed[top:top + P, left:left + P]
                allow_ratio = float(p_allowed.mean())
                if allow_ratio < self.MIN_ALLOWED_RATIO_IN_PATCH:
                    continue

                save_name = f"{line_stem}_NEG_sed_r{top}_c{left}.npz"
                meta = {
                    "orig_file": line_stem,
                    "neg_type": "sediment_uniform_250_2500_outside_pos",
                    "top": int(top),
                    "left": int(left),
                    "allow_ratio": float(allow_ratio),
                    "pos_footprint_used_patches": int(pos_used),
                    "pos_footprint_buffer": int(self.POS_FOOTPRINT_BUFFER),
                    "sed_ms0": int(self.SED_MS0),
                    "sed_ms1": int(self.SED_MS1),
                    "saved_keys": sorted([k for k in arrs.keys()]),
                }
                self._save_patch(arrs, os.path.join(self.dir_sed, save_name), meta)
                used_coords.add(key)
                saved_sed += 1

        # -----------------------------
        # 2) Shallow (0~250): sparse random
        # -----------------------------
        saved_wat = 0
        if self.N_SHALLOW_PER_LINE > 0 and np.any(shallow_allowed):
            ys, xs = np.where(shallow_allowed)
            for i in range(self.N_SHALLOW_PER_LINE):
                idx = np.random.randint(len(ys))
                rr, cc = int(ys[idx]), int(xs[idx])

                top = rr - P // 2 + np.random.randint(-self.RAND_JITTER, self.RAND_JITTER + 1)
                left = cc - P // 2 + np.random.randint(-self.RAND_JITTER, self.RAND_JITTER + 1)
                top, left = self._clamp_top_left(top, left, H, W)

                key = (top, left)
                if self.DEDUP and key in used_coords:
                    continue
                if self.AVOID_POS_FOOTPRINT_FOR_WATER_DEEP and np.any(pos_footprint[top:top + P, left:left + P]):
                    continue

                p_msk = mask[top:top + P, left:left + P]
                if self._valid_ratio(p_msk) < self.MIN_VALID_RATIO:
                    continue
                if self._pos_pixels(p_msk) > 0:
                    continue

                p_allowed = shallow_allowed[top:top + P, left:left + P]
                if float(p_allowed.mean()) < self.MIN_ALLOWED_RATIO_IN_PATCH:
                    continue

                save_name = f"{line_stem}_NEG_water_i{i}_r{top}_c{left}.npz"
                meta = {
                    "orig_file": line_stem,
                    "neg_type": "shallow_0_250_sparse",
                    "top": int(top),
                    "left": int(left),
                    "shallow_ms0": int(self.SHALLOW_MS0),
                    "shallow_ms1": int(self.SHALLOW_MS1),
                    "pos_footprint_used_patches": int(pos_used),
                    "pos_footprint_buffer": int(self.POS_FOOTPRINT_BUFFER),
                    "saved_keys": sorted([k for k in arrs.keys()]),
                }
                self._save_patch(arrs, os.path.join(self.dir_wat, save_name), meta)
                used_coords.add(key)
                saved_wat += 1

        # -----------------------------
        # 3) Deep (>2500): sparse random
        # -----------------------------
        saved_dep = 0
        if self.N_DEEP_PER_LINE > 0 and np.any(deep_allowed):
            ys, xs = np.where(deep_allowed)
            for i in range(self.N_DEEP_PER_LINE):
                idx = np.random.randint(len(ys))
                rr, cc = int(ys[idx]), int(xs[idx])

                top = rr - P // 2 + np.random.randint(-self.RAND_JITTER, self.RAND_JITTER + 1)
                left = cc - P // 2 + np.random.randint(-self.RAND_JITTER, self.RAND_JITTER + 1)
                top, left = self._clamp_top_left(top, left, H, W)

                key = (top, left)
                if self.DEDUP and key in used_coords:
                    continue
                if self.AVOID_POS_FOOTPRINT_FOR_WATER_DEEP and np.any(pos_footprint[top:top + P, left:left + P]):
                    continue

                p_msk = mask[top:top + P, left:left + P]
                if self._valid_ratio(p_msk) < self.MIN_VALID_RATIO:
                    continue
                if self._pos_pixels(p_msk) > 0:
                    continue

                p_allowed = deep_allowed[top:top + P, left:left + P]
                if float(p_allowed.mean()) < self.MIN_ALLOWED_RATIO_IN_PATCH:
                    continue

                save_name = f"{line_stem}_NEG_deep_i{i}_r{top}_c{left}.npz"
                meta = {
                    "orig_file": line_stem,
                    "neg_type": "deep_below_2500_sparse",
                    "top": int(top),
                    "left": int(left),
                    "deep_ms_start": int(self.DEEP_MS_START),
                    "pos_footprint_used_patches": int(pos_used),
                    "pos_footprint_buffer": int(self.POS_FOOTPRINT_BUFFER),
                    "saved_keys": sorted([k for k in arrs.keys()]),
                }
                self._save_patch(arrs, os.path.join(self.dir_dep, save_name), meta)
                used_coords.add(key)
                saved_dep += 1

        return saved_sed, saved_wat, saved_dep, pos_used, pos_skipped_other

    def run(self):
        files = sorted(glob.glob(os.path.join(self.line_npz_dir, "*.npz")))
        if not files:
            print(f"No line NPZ files found in: {self.line_npz_dir}")
            return

        print(f"Negative sampling | lines={len(files)}")
        print(f"   PATCH={self.PATCH} | DT_MS={self.DT_MS}")
        print(f"   Shallow: {self.SHALLOW_MS0}-{self.SHALLOW_MS1}ms | N={self.N_SHALLOW_PER_LINE}")
        print(f"   Sediment: {self.SED_MS0}-{self.SED_MS1}ms | strideT/X={self.SED_STRIDE_T}/{self.SED_STRIDE_X}")
        print(f"   Deep: >{self.DEEP_MS_START}ms | N={self.N_DEEP_PER_LINE}")
        print(f"   SED_REQUIRE_NO_POS_OVERLAP={self.SED_REQUIRE_NO_POS_OVERLAP} | POS_FOOTPRINT_BUFFER={self.POS_FOOTPRINT_BUFFER}")
        print(f"   OUTPUT: {self.dir_neg}")

        total_sed = total_wat = total_dep = 0
        total_pos_used = 0

        for fp in tqdm(files, desc="Lines"):
            try:
                s, w, d, pos_used, pos_skip = self._process_one_line(fp)
                total_sed += s
                total_wat += w
                total_dep += d
                total_pos_used += pos_used
            except Exception as e:
                print(f"\nError: {os.path.basename(fp)} - {e}")

        print("\nDone.")
        print(f"   Sediment saved: {total_sed} -> {self.dir_sed}")
        print(f"   Shallow  saved: {total_wat} -> {self.dir_wat}")
        print(f"   Deep     saved: {total_dep} -> {self.dir_dep}")
        print(f"   POS patches used (sum): {total_pos_used}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract stratified negative patches around positive training examples."
    )
    parser.add_argument("--line-dir", required=True, help="Directory containing attributed line NPZ files")
    parser.add_argument("--positive-dir", required=True, help="Directory containing positive patch NPZ files")
    parser.add_argument("--output-dir", required=True, help="Directory for negative patches")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen = StratifiedNegAroundPosSampler(
        line_npz_dir=args.line_dir,
        pos_patch_dir=args.positive_dir,
        output_root=args.output_dir,
        seed=args.seed
    )

    # ---- You can tune these ----
    gen.DT_MS = 1.0

    gen.SHALLOW_MS0 = 0
    gen.SHALLOW_MS1 = 250
    gen.SED_MS0 = 250
    gen.SED_MS1 = 2500
    gen.DEEP_MS_START = 2500

    gen.N_SHALLOW_PER_LINE = 18   # Sparse sampling.
    gen.N_DEEP_PER_LINE = 22      # Sparse sampling.

    gen.SED_STRIDE_T = 128        # Coverage density.
    gen.SED_STRIDE_X = 128

    gen.SED_REQUIRE_NO_POS_OVERLAP = True
    gen.MIN_ALLOWED_RATIO_IN_PATCH = 1.0

    gen.POS_FOOTPRINT_BUFFER = 0  # Increase to 16 or 32 to separate negatives.
    gen.AVOID_POS_FOOTPRINT_FOR_WATER_DEEP = True

    gen.run()
