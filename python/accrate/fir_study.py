"""Long FIR filtering with each accumulator: output SNR without requantization.

Signal: grayscale Kodak images, rows concatenated, centred to signed 8-bit.
Filter: K-tap windowed-sinc low-pass, quantized to signed 8-bit taps.
Each output accumulates K signed products serially in the design's 24-bit
accumulator (bit-exact low-part tables).  Drift compensation preloads
-round(E E_K) from Theorem 3 under the product-residue law measured on
images 1-12; SNR is measured on images 13-24 (held out), against the exact
FIR output.  Also reports the Theorem 3 prediction of the error MSE.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from cnn_accumulate import Accumulator  # noqa: E402
from theory import low_tables, moments  # noqa: E402

HERE = Path(__file__).parent
KODAK = HERE.parent / "data" / "kodak"


def load(ids):
    xs = [np.asarray(Image.open(KODAK / f"kodim{i:02d}.png").convert("L"), dtype=np.int64).ravel() - 128
          for i in ids]
    return np.concatenate(xs)


def taps(K):
    n = np.arange(K) - (K - 1) / 2
    h = np.sinc(n / 4) * np.hanning(K)
    return np.round(h / np.abs(h).max() * 127).astype(np.int64)


def windows(x, K, count, rng):
    idx = rng.integers(0, len(x) - K, count)
    return np.stack([x[i:i + K] for i in idx])


def main(K=256, count=20000):
    rng = np.random.default_rng(5)
    h = taps(K)
    Xtr = windows(load(range(1, 13)), K, count, rng)
    Xte = windows(load(range(13, 25)), K, count, rng)
    hist = np.bincount(((Xtr[:2000] * h[None, :]) & 255).ravel(), minlength=256).astype(float)
    exact = (Xte * h[None, :]).sum(1)
    sig = float(np.mean(exact.astype(float) ** 2))
    designs = (["exact"] + [f"round_{t}" for t in range(1, 9)] + [f"trunc_{t}" for t in range(1, 9)] +
               ["evo_add8u_6QU", "evo_add8u_0E2", "eta1_5", "loawa_4", "loa_5", "ama5_4", "evo_add8u_6SM",
                "evo_add8u_0NS", "evo_add8u_6MZ", "evo_add8u_2LL", "ama2_3", "evo_add8u_6S4", "evo_add8u_0NQ",
                "evo_add8u_0PL", "evo_add8u_0TP", "eta1_4", "loa_4", "evo_add8u_4M7"])
    rows = []
    for d in designs:
        pre, pred = 0, 0.0
        if d != "exact":
            k, z, c = low_tables(d)
            pmf = hist.reshape(-1, 1 << k).sum(0)
            mean, var = moments(z, c, [K], pmf=pmf)[K]
            pre = int(-round(mean))
            pred = var + (mean + pre) ** 2
        acc = Accumulator(d, "calibrated", {"fir": pre})
        acc.layer = "fir"
        P = torch.from_numpy(Xte * h[None, :])
        out = acc.run(P[:, j] for j in range(K)).numpy()
        err = (out - exact).astype(float)
        mse = float(np.mean(err ** 2))
        snr = 10 * np.log10(sig / mse) if mse > 0 else float("inf")
        # bootstrap over windows (1000 resamples) for a 95% SNR interval
        brng = np.random.default_rng(1)
        e2, s2 = err ** 2, exact.astype(float) ** 2
        bs = []
        for _ in range(1000):
            i = brng.integers(0, len(e2), len(e2))
            bs.append(10 * np.log10(s2[i].mean() / e2[i].mean()) if e2[i].mean() > 0 else np.inf)
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        rows.append({"design": d, "K": K, "mse": mse, "pred_mse": pred, "snr_db": snr, "snr_ci95": ci,
                     "preload": pre})
        print(f"{d:16s} MSE {mse:12.2f} pred {pred:12.2f} SNR {snr:7.2f} dB", flush=True)
    (HERE / f"fir_study_K{K}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
