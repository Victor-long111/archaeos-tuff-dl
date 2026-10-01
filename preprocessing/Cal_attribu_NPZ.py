import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class NPZAttrGenerator:
    """
    Input NPZ:
      - raw: (Time, Trace) 2D
      - mask(optional): (Time, Trace)

    Output NPZ:
      - raw(optional)
      - mask(optional)
      - env : envelope(Time, Trace)  (log1p optional)
      - rms : moving RMS(Time, Trace) along Time axis
      - ifreq: instantaneous frequency(Time, Trace) along Time axis (Hz if dt_ms provided)
      - cosp: cos(instantaneous phase)
      - sinp: sin(instantaneous phase)
      - coh : coherence-lite (local similarity between neighboring traces, 0..1)
    """

    def __init__(
        self,
        npz_dir,
        output_dir,
        axis_tag="Time,Trace",
        keep_raw=True,
        keep_mask=True,
        overwrite=True,

        # Envelope
        env_log1p=True,

        # RMS
        rms_win=21,          # window length in samples (odd recommended)
        rms_log1p=False,

        # Instantaneous frequency
        dt_ms=1.0,           # Default seismic sampling interval: 1 ms.
        ifreq_smooth_win=5,  # moving average smoothing (>=1)
        ifreq_gate=True,     # gate low-amplitude area using env threshold
        ifreq_gate_pct=10,   # threshold = percentile(env_raw, pct)
        ifreq_fill_value=0.0,
        ifreq_clip_hz=250.0, # Clip unstable instantaneous-frequency spikes.

        # Coherence-lite (local similarity)
        coh_enable=True,
        coh_win_t=21,        # time-window length for correlation (odd recommended)
        coh_neighbor_k=1,    # neighbor distance in traces (1 = adjacent)
        coh_smooth_t=5,      # optional smoothing along time
        coh_smooth_x=3,      # optional smoothing along trace
        coh_eps=1e-6,

        qc_show=2,
        seed=42
    ):
        self.npz_dir = npz_dir
        self.output_dir = output_dir
        self.axis_tag = axis_tag
        self.keep_raw = keep_raw
        self.keep_mask = keep_mask
        self.overwrite = overwrite

        self.env_log1p = env_log1p

        self.rms_win = int(rms_win)
        self.rms_log1p = rms_log1p

        self.dt_ms = dt_ms
        self.ifreq_smooth_win = int(ifreq_smooth_win)
        self.ifreq_gate = bool(ifreq_gate)
        self.ifreq_gate_pct = float(ifreq_gate_pct)
        self.ifreq_fill_value = float(ifreq_fill_value)
        self.ifreq_clip_hz = float(ifreq_clip_hz) if ifreq_clip_hz is not None else None

        self.coh_enable = bool(coh_enable)
        self.coh_win_t = int(coh_win_t)
        self.coh_neighbor_k = int(coh_neighbor_k)
        self.coh_smooth_t = int(coh_smooth_t)
        self.coh_smooth_x = int(coh_smooth_x)
        self.coh_eps = float(coh_eps)

        self.qc_show = int(qc_show)
        random.seed(seed)
        np.random.seed(seed)

        os.makedirs(self.output_dir, exist_ok=True)

    # -----------------------------
    # Robust key loader
    # -----------------------------
    def _guess_key(self, keys, keywords):
        for k in keys:
            kl = k.lower()
            if any(kw in kl for kw in keywords):
                return k
        return None

    def _load_npz(self, fp):
        d = np.load(fp, allow_pickle=True)
        keys = list(d.files)
        raw_k = self._guess_key(keys, ("raw", "seis", "amp", "amplitude", "img", "data"))
        msk_k = self._guess_key(keys, ("mask", "label", "gt", "msk"))

        if raw_k is None:
            raise KeyError(f"[{fp}] cannot find raw key in {keys}")

        raw = np.squeeze(d[raw_k]).astype(np.float32)
        mask = np.squeeze(d[msk_k]).astype(np.int16) if (msk_k is not None) else None

        if raw.ndim != 2:
            raise ValueError(f"[{fp}] raw must be 2D (Time,Trace), got {raw.shape}")
        if mask is not None and mask.shape != raw.shape:
            raise ValueError(f"[{fp}] mask shape mismatch {mask.shape} vs raw {raw.shape}")

        return raw, mask

    def _axis_from_tag(self):
        tag = self.axis_tag.replace(" ", "").lower()
        return 0 if tag.startswith("time") else 1

    # -----------------------------
    # Hilbert analytic signal by FFT (no scipy)
    # -----------------------------
    def _analytic_signal_fft(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        x = x.astype(np.float32)
        N = x.shape[axis]
        Xf = np.fft.fft(x, axis=axis)

        h = np.zeros(N, dtype=np.float32)
        if N % 2 == 0:
            h[0] = 1.0
            h[N // 2] = 1.0
            h[1:N // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(N + 1) // 2] = 2.0

        shape = [1] * x.ndim
        shape[axis] = N
        h = h.reshape(shape)

        Xa = Xf * h
        xa = np.fft.ifft(Xa, axis=axis)  # complex
        return xa

    def _envelope_phase_pair(self, raw: np.ndarray, axis: int = 0):
        """
        Returns:
          - env_raw: absolute analytic signal used for amplitude gating.
          - env_out: envelope output, optionally transformed with log1p.
          - phase  : angle(analytic) (float32)
        """
        xa = self._analytic_signal_fft(raw, axis=axis)
        env_raw = np.abs(xa).astype(np.float32)
        env_out = np.log1p(env_raw) if self.env_log1p else env_raw
        phase = np.angle(xa).astype(np.float32)
        return env_raw, env_out, phase

    # -----------------------------
    # Moving RMS along Time axis (axis)
    # -----------------------------
    def _moving_rms(self, raw: np.ndarray, axis: int = 0, win: int = 21) -> np.ndarray:
        if win <= 1:
            return np.abs(raw).astype(np.float32)

        win = int(win)
        if win % 2 == 0:
            win += 1

        x2 = (raw.astype(np.float32) ** 2)
        x2m = np.moveaxis(x2, axis, 0)  # (T, ...)
        pad = win // 2

        xpad = np.pad(
            x2m,
            ((pad, pad),) + ((0, 0),) * (x2m.ndim - 1),
            mode="edge"
        )  # (T+2pad, ...)

        c = np.cumsum(xpad, axis=0)
        z = np.zeros_like(c[:1])
        c0 = np.concatenate([z, c], axis=0)  # (T+2pad+1, ...)

        wsum = c0[win:] - c0[:-win]  # (T, ...)
        mean = wsum / float(win)
        rmsm = np.sqrt(np.maximum(mean, 0.0)).astype(np.float32)

        rms = np.moveaxis(rmsm, 0, axis)
        if self.rms_log1p:
            rms = np.log1p(rms)
        return rms

    # -----------------------------
    # moving average
    # -----------------------------
    def _moving_average(self, x: np.ndarray, axis: int = 0, win: int = 5) -> np.ndarray:
        if win <= 1:
            return x.astype(np.float32)

        win = int(win)
        if win % 2 == 0:
            win += 1

        xm = np.moveaxis(x.astype(np.float32), axis, 0)  # (T, ...)
        pad = win // 2

        xpad = np.pad(
            xm,
            ((pad, pad),) + ((0, 0),) * (xm.ndim - 1),
            mode="edge"
        )  # (T+2pad, ...)

        c = np.cumsum(xpad, axis=0)
        z = np.zeros_like(c[:1])
        c0 = np.concatenate([z, c], axis=0)

        wsum = c0[win:] - c0[:-win]          # (T, ...)
        out = (wsum / float(win)).astype(np.float32)
        return np.moveaxis(out, 0, axis)

    # -----------------------------
    # Moving sum (for coherence-lite)
    # -----------------------------
    def _moving_sum(self, x: np.ndarray, axis: int = 0, win: int = 21) -> np.ndarray:
        """
        Moving sum along given axis with edge padding, output same shape as input.
        """
        if win <= 1:
            return x.astype(np.float32)

        win = int(win)
        if win % 2 == 0:
            win += 1

        xm = np.moveaxis(x.astype(np.float32), axis, 0)  # (T, ...)
        pad = win // 2

        xpad = np.pad(
            xm,
            ((pad, pad),) + ((0, 0),) * (xm.ndim - 1),
            mode="edge"
        )

        c = np.cumsum(xpad, axis=0)
        z = np.zeros_like(c[:1])
        c0 = np.concatenate([z, c], axis=0)

        wsum = c0[win:] - c0[:-win]          # same length as original
        out = np.moveaxis(wsum, 0, axis)
        return out.astype(np.float32)

    # -----------------------------
    # Instantaneous frequency (from phase, no extra Hilbert)
    # -----------------------------
    def _instantaneous_frequency(self, phase: np.ndarray, env_raw: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        ifreq = (1 / 2*pi) dphi/dt
        The analytic-signal phase is differentiated with a centered difference
        and returned in Hz. The default sampling interval is 1 ms.
        """
        phm = np.moveaxis(phase, axis, 0)      # (T,...)
        phu = np.unwrap(phm, axis=0).astype(np.float32)

        dphi = np.zeros_like(phu, dtype=np.float32)
        dphi[1:-1] = 0.5 * (phu[2:] - phu[:-2])
        dphi[0] = phu[1] - phu[0]
        dphi[-1] = phu[-1] - phu[-2]

        ifreq_m = (dphi / (2.0 * np.pi)).astype(np.float32)

        if self.dt_ms is not None:
            dt_s = float(self.dt_ms) / 1000.0
            ifreq_m = ifreq_m / max(dt_s, 1e-12)

        ifreq = np.moveaxis(ifreq_m, 0, axis)

        if self.ifreq_smooth_win > 1:
            ifreq = self._moving_average(ifreq, axis=axis, win=self.ifreq_smooth_win)

        if self.ifreq_gate:
            thr = np.percentile(env_raw, self.ifreq_gate_pct)
            gate = (env_raw <= thr)
            ifreq = ifreq.copy()
            ifreq[gate] = self.ifreq_fill_value

        if (self.ifreq_clip_hz is not None) and (self.dt_ms is not None):
            ifreq = np.clip(ifreq, -self.ifreq_clip_hz, self.ifreq_clip_hz).astype(np.float32)

        return ifreq.astype(np.float32)

    # -----------------------------
    # Coherence-lite (local similarity)
    # -----------------------------
    def _coherence_lite(self, raw: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        coherence-lite: windowed normalized correlation between neighboring traces.
        Output: [0,1] (abs correlation).

        corr(i, i+k) = |sum(a*b)| / sqrt(sum(a^2)*sum(b^2) + eps)
        sums computed in moving time window.
        """
        win = int(self.coh_win_t)
        if win % 2 == 0:
            win += 1

        K = max(1, int(self.coh_neighbor_k))
        eps = float(self.coh_eps)

        x = np.moveaxis(raw.astype(np.float32), axis, 0)  # (T, W)
        if x.ndim != 2:
            raise ValueError(f"coherence-lite expects 2D, got {x.shape}")

        T, W = x.shape
        coh = np.zeros((T, W), dtype=np.float32)
        cnt = np.zeros((T, W), dtype=np.float32)

        for k in range(1, K + 1):
            if W - k <= 0:
                break

            a = x[:, :-k]
            b = x[:, k:]

            sum_ab = self._moving_sum(a * b, axis=0, win=win)      # (T, W-k)
            sum_a2 = self._moving_sum(a * a, axis=0, win=win)
            sum_b2 = self._moving_sum(b * b, axis=0, win=win)

            denom = np.sqrt(np.maximum(sum_a2 * sum_b2, 0.0) + eps)
            corr = np.abs(sum_ab) / denom
            corr = np.clip(corr, 0.0, 1.0).astype(np.float32)

            coh[:, :-k] += corr
            coh[:, k:]  += corr
            cnt[:, :-k] += 1.0
            cnt[:, k:]  += 1.0

        coh = np.where(cnt > 0, coh / cnt, 0.0).astype(np.float32)

        # optional smoothing (time then trace)
        if self.coh_smooth_t and self.coh_smooth_t > 1:
            coh = self._moving_average(coh, axis=0, win=self.coh_smooth_t)
        if self.coh_smooth_x and self.coh_smooth_x > 1:
            coh = self._moving_average(coh, axis=1, win=self.coh_smooth_x)

        return coh.astype(np.float32)

    # -----------------------------
    # Run
    # -----------------------------
    def run(self):
        print("Start batch attribute generation (env/rms/ifreq/cosp/sinp/coh)...")
        files = sorted(glob.glob(os.path.join(self.npz_dir, "*.npz")))
        print(f"Found {len(files)} NPZ file(s)")
        if not files:
            print("No NPZ files found")
            return

        axis = self._axis_from_tag()
        print(f"axis_tag={self.axis_tag} -> time axis={axis}")
        print(f"dt_ms={self.dt_ms} (Hz output)")
        print(f"env_log1p={self.env_log1p} | rms_win={self.rms_win}")
        print(f"ifreq smooth={self.ifreq_smooth_win} gate={self.ifreq_gate} gate_pct={self.ifreq_gate_pct} clip={self.ifreq_clip_hz}")
        print(f"coh_enable={self.coh_enable} | coh_win_t={self.coh_win_t} k={self.coh_neighbor_k} smooth(t,x)=({self.coh_smooth_t},{self.coh_smooth_x})")

        success = 0
        for fp in tqdm(files, desc="Attr"):
            try:
                raw, mask = self._load_npz(fp)

                # Compute the analytic signal once for envelope and phase.
                env_raw, env, phase = self._envelope_phase_pair(raw, axis=axis)

                # Recommended phase-derived channels.
                cosp = np.cos(phase).astype(np.float32)
                sinp = np.sin(phase).astype(np.float32)

                rms = self._moving_rms(raw, axis=axis, win=self.rms_win)
                ifreq = self._instantaneous_frequency(phase=phase, env_raw=env_raw, axis=axis)

                coh = None
                if self.coh_enable:
                    coh = self._coherence_lite(raw, axis=axis)

                base = os.path.splitext(os.path.basename(fp))[0]
                out_fp = os.path.join(self.output_dir, f"{base}.npz")

                if (not self.overwrite) and os.path.exists(out_fp):
                    continue

                save_dict = {
                    "env": env.astype(np.float32),
                    "rms": rms.astype(np.float32),
                    "ifreq": ifreq.astype(np.float32),
                    "cosp": cosp,
                    "sinp": sinp,
                    "meta_attr": np.array([{
                        "axis_tag": self.axis_tag,
                        "dt_ms": self.dt_ms,
                        "env_log1p": self.env_log1p,
                        "rms_win": self.rms_win,
                        "rms_log1p": self.rms_log1p,
                        "ifreq_smooth_win": self.ifreq_smooth_win,
                        "ifreq_gate": self.ifreq_gate,
                        "ifreq_gate_pct": self.ifreq_gate_pct,
                        "ifreq_fill_value": self.ifreq_fill_value,
                        "ifreq_clip_hz": self.ifreq_clip_hz,
                        "added_attrs": ["cosp", "sinp", "coh"] if self.coh_enable else ["cosp", "sinp"],
                        "coh_enable": self.coh_enable,
                        "coh_win_t": self.coh_win_t,
                        "coh_neighbor_k": self.coh_neighbor_k,
                        "coh_smooth_t": self.coh_smooth_t,
                        "coh_smooth_x": self.coh_smooth_x,
                        "coh_eps": self.coh_eps,
                    }], dtype=object)
                }

                if coh is not None:
                    save_dict["coh"] = coh.astype(np.float32)

                if self.keep_raw:
                    save_dict["raw"] = raw.astype(np.float32)
                if self.keep_mask and (mask is not None):
                    save_dict["mask"] = mask

                np.savez_compressed(out_fp, **save_dict)
                success += 1

            except Exception as e:
                print(f"\n   Failed {os.path.basename(fp)}: {e}")

        print(f"\nDone. Generated {success} attribute NPZ file(s).")
        print(f"Output directory: {self.output_dir}")

        self.inspect_generated_files(n_show=self.qc_show)

    # -----------------------------
    # QC Visualize
    # -----------------------------
    def inspect_generated_files(self, n_show=2):
        print("\nInspecting a random generated result...")
        files = glob.glob(os.path.join(self.output_dir, "*.npz"))
        if not files:
            print("No NPZ files found in the output directory")
            return

        samples = random.sample(files, min(n_show, len(files)))

        for fp in samples:
            d = np.load(fp, allow_pickle=True)
            env = d["env"]
            rms = d["rms"]
            ifreq = d["ifreq"]
            cosp = d["cosp"] if "cosp" in d.files else None
            sinp = d["sinp"] if "sinp" in d.files else None
            coh = d["coh"] if "coh" in d.files else None
            raw = d["raw"] if "raw" in d.files else None
            mask = d["mask"] if "mask" in d.files else None

            title = os.path.basename(fp) + f" | shape={env.shape}"

            # crop by mask activity
            if mask is not None:
                active_traces = np.where(np.any(mask > 0, axis=0))[0]
                active_times = np.where(np.any(mask > 0, axis=1))[0]
                if len(active_traces) and len(active_times):
                    c0 = max(0, active_traces.min() - 50)
                    c1 = min(env.shape[1], active_traces.max() + 50)
                    r0 = max(0, active_times.min() - 100)
                    r1 = min(env.shape[0], active_times.max() + 100)
                else:
                    r0, r1, c0, c1 = 0, env.shape[0], 0, env.shape[1]
            else:
                r0, r1, c0, c1 = 0, env.shape[0], 0, env.shape[1]

            def show_img(ax, img, cmap="gray", vmin=None, vmax=None, m=None):
                ax.imshow(img, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
                if m is not None:
                    ax.imshow(m, cmap="Reds", alpha=0.45, interpolation="none", aspect="auto")
                ax.set_xlabel("Trace")
                ax.set_ylabel("Time")

            raw_show = raw[r0:r1, c0:c1] if raw is not None else None
            env_show = env[r0:r1, c0:c1]
            rms_show = rms[r0:r1, c0:c1]
            ifr_show = ifreq[r0:r1, c0:c1]
            m_show = mask[r0:r1, c0:c1] if mask is not None else None

            # Main 2x2
            plt.figure(figsize=(18, 10))

            if raw_show is not None:
                ax = plt.subplot(2, 2, 1)
                v = np.std(raw_show) * 2.0 + 1e-6
                show_img(ax, raw_show, vmin=-v, vmax=v, m=m_show)
                ax.set_title("RAW (+GT)")

            ax = plt.subplot(2, 2, 2)
            vmax_env = np.percentile(env_show, 99.5)
            show_img(ax, env_show, vmin=0, vmax=vmax_env, m=m_show)
            ax.set_title("ENVELOPE (+GT)")

            ax = plt.subplot(2, 2, 3)
            vmax_rms = np.percentile(rms_show, 99.5)
            show_img(ax, rms_show, vmin=0, vmax=vmax_rms, m=m_show)
            ax.set_title(f"RMS(win={self.rms_win}) (+GT)")

            ax = plt.subplot(2, 2, 4)
            p1, p99 = np.percentile(ifr_show, 1.0), np.percentile(ifr_show, 99.0)
            show_img(ax, ifr_show, vmin=p1, vmax=p99, m=m_show)
            ax.set_title("Instantaneous Frequency (Hz) (+GT)")

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()

            # Extra QC: phase + coherence
            if cosp is not None and sinp is not None:
                plt.figure(figsize=(18, 6))
                ax1 = plt.subplot(1, 2, 1)
                show_img(ax1, cosp[r0:r1, c0:c1], vmin=-1, vmax=1, m=m_show)
                ax1.set_title("cos(phase) (+GT)")
                ax2 = plt.subplot(1, 2, 2)
                show_img(ax2, sinp[r0:r1, c0:c1], vmin=-1, vmax=1, m=m_show)
                ax2.set_title("sin(phase) (+GT)")
                plt.tight_layout()
                plt.show()

            if coh is not None:
                plt.figure(figsize=(18, 6))
                ax = plt.subplot(1, 1, 1)
                show_img(ax, coh[r0:r1, c0:c1], vmin=0, vmax=1, m=m_show)
                ax.set_title("Coherence-lite (0..1) (+GT)")
                plt.tight_layout()
                plt.show()

            # shape asserts
            if raw is not None:
                assert env.shape == raw.shape
                assert rms.shape == raw.shape
                assert ifreq.shape == raw.shape
                assert cosp.shape == raw.shape
                assert sinp.shape == raw.shape
                if coh is not None:
                    assert coh.shape == raw.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute seismic attributes for line-level NPZ files."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input NPZ files")
    parser.add_argument("--output-dir", required=True, help="Directory for attributed NPZ files")
    parser.add_argument("--qc-show", type=int, default=0, help="Number of interactive QC examples")
    args = parser.parse_args()

    gen = NPZAttrGenerator(
        npz_dir=args.input_dir,
        output_dir=args.output_dir,
        axis_tag="Time,Trace",

        keep_raw=True,
        keep_mask=True,
        overwrite=True,

        env_log1p=True,
        rms_win=21,
        rms_log1p=False,

        dt_ms=1.0,
        ifreq_smooth_win=5,
        ifreq_gate=True,
        ifreq_gate_pct=10,
        ifreq_fill_value=0.0,
        ifreq_clip_hz=250.0,

        coh_enable=True,
        coh_win_t=21,
        coh_neighbor_k=1,
        coh_smooth_t=5,
        coh_smooth_x=3,
        coh_eps=1e-6,

        qc_show=args.qc_show
    )
    gen.run()
