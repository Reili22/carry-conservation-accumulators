"""Bit-exact INT8 ResNet-20 inference with approximate serial accumulators.

BN is folded into each conv; weights are symmetric per-output-channel INT8 and
activations symmetric per-tensor INT8 (scales calibrated on training images).
Every conv/FC output is accumulated serially over K = C_in*kh*kw products in a
W-bit two's-complement register with the design's low-part tables:
    T <- (((T >> k) + (U >> k) + c[x, u]) << k | z[x, u]) mod 2^W,
x = T mod 2^k, u = U mod 2^k; identical to the verified RTL by the residue
property.  Optional drift compensation preloads -round(K * drift) at clear.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from resnet20 import CKPT, CLASSES, ResNet20, load_cifar  # noqa: E402
from theory import low_tables, metrics, moments  # noqa: E402

W = 24
MASKW = (1 << W) - 1


def fold(conv, bn):
    w = conv.weight.detach()
    s = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    return w * s[:, None, None, None], bn.bias.detach() - bn.running_mean * s


def quant_w(w):
    s = w.abs().flatten(1).amax(1).clamp(min=1e-12) / 127.0
    return torch.round(w / s.view(-1, *[1] * (w.dim() - 1))).clamp(-127, 127), s


class Accumulator:
    """compensate: "none"; "uniform" (preload -round(K*drift) with the uniform-law drift);
    or "calibrated" (preload -round(E[E_K]) from Theorem 3 under each layer's residue law
    measured on calibration images, given in `preloads`)."""

    def __init__(self, design, compensate="none", preloads=None):
        self.design = design
        self.mode = compensate
        self.preloads = preloads or {}
        self.layer = None
        if design == "exact":
            self.k = 0
        else:
            k, z, c = low_tables(design)
            self.k = k
            self.z = torch.from_numpy(z.ravel().astype(np.int64))
            self.c = torch.from_numpy(c.ravel().astype(np.int64))
        self.drift = metrics(design)["drift"] if (compensate == "uniform" and design != "exact") else 0.0

    def preload(self):
        if self.mode == "calibrated":
            return self.preloads.get(self.layer, 0)
        return int(-round(self._K * self.drift))

    def run(self, prods):
        """prods: iterator over K tensors of signed int64 products (same shape)."""
        T = None
        K = 0
        for p in prods:
            U = p & MASKW
            if T is None:
                T = torch.full_like(U, self.preload() & MASKW)
            if self.k == 0:
                T = (T + U) & MASKW
            else:
                q = 1 << self.k
                idx = (T & (q - 1)) * q + (U & (q - 1))
                T = ((((T >> self.k) + (U >> self.k) + self.c[idx]) << self.k) | self.z[idx]) & MASKW
            K += 1
        return torch.where(T >= (1 << (W - 1)), T - (1 << W), T)


class QNet:
    """Integer re-implementation of the trained ResNet-20 forward pass."""

    def __init__(self, model, calib):
        self.layers = {}
        convs = [("conv1", model.conv1, model.bn1)]
        for i, b in enumerate(model.layers):
            convs += [(f"l{i}a", b.conv1, b.bn1), (f"l{i}b", b.conv2, b.bn2)]
        for name, conv, bn in convs:
            w, bias = fold(conv, bn)
            wq, sw = quant_w(w)
            self.layers[name] = dict(wq=wq.long(), sw=sw, bias=bias, stride=conv.stride[0])
        wq, sw = quant_w(model.fc.weight.detach())
        self.layers["fc"] = dict(wq=wq.long(), sw=sw, bias=model.fc.bias.detach())
        self.model = model
        self.act_scale = {}
        self._calibrate(calib)

    def _calibrate(self, x):
        self.act_scale = {}
        self.forward(x, None, calibrate=True)

    def _qa(self, name, x, calibrate):
        if calibrate:
            self.act_scale[name] = max(self.act_scale.get(name, 0.0), float(x.abs().quantile(0.9999)) / 127.0)
        s = self.act_scale[name]
        return torch.round(x / s).clamp(-127, 127).long(), s

    def conv(self, name, x, acc, calibrate, stats):
        L = self.layers[name]
        xq, sa = self._qa(name, x, calibrate)
        B, C, H, Wd = x.shape
        stride = L.get("stride", 1)
        cols = F.unfold(xq.double(), 3, padding=1, stride=stride).long()  # (B, K, Lout)
        wmat = L["wq"].flatten(1)  # (Cout, K)
        Ho, Wo = (H - 1) // stride + 1, (Wd - 1) // stride + 1
        if calibrate or acc is None:
            out = torch.einsum("bkl,ck->bcl", cols.double(), wmat.double()).round().long()
        else:
            acc.layer = name
            out = acc.run(cols[:, j, None, :] * wmat[None, :, j, None] for j in range(cols.shape[1]))
            if stats is not None:
                exact = torch.einsum("bkl,ck->bcl", cols.double(), wmat.double()).round().long()
                e = (out - exact).double()
                st = stats.setdefault(name, {"n": 0, "s1": 0.0, "s2": 0.0, "K": cols.shape[1],
                                             "hist256": np.zeros(256)})
                st["n"] += e.numel(); st["s1"] += e.sum().item(); st["s2"] += (e * e).sum().item()
                # operand residue law (mod 256) from a subsample of this layer's products
                p = (cols[:4, :, None, :] * wmat.T[None, :, :, None]) & 255
                st["hist256"] += np.bincount(p.flatten().numpy(), minlength=256)
        y = out.double() * sa * L["sw"].double()[None, :, None] + L["bias"].double()[None, :, None]
        return y.view(B, -1, Ho, Wo).float()

    def forward(self, x, acc, calibrate=False, stats=None):
        y = F.relu(self.conv("conv1", x, acc, calibrate, stats))
        for i, b in enumerate(self.model.layers):
            h = F.relu(self.conv(f"l{i}a", y, acc, calibrate, stats))
            h = self.conv(f"l{i}b", h, acc, calibrate, stats)
            y = F.relu(h + b.shortcut(y))
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        L = self.layers["fc"]
        xq, sa = self._qa("fc", y, calibrate)
        if calibrate or acc is None:
            out = (xq @ L["wq"].T)
        else:
            acc.layer = "fc"
            out = acc.run(xq[:, j, None] * L["wq"][None, :, j] for j in range(xq.shape[1]))
        return out.double() * sa * L["sw"].double()[None, :] + L["bias"].double()[None, :]


def residue_histograms(qnet, x):
    """Per-layer histogram of product residues mod 256 (and K) on calibration images, exact arithmetic."""
    hist = {}

    def collect(name, cols, w):
        h = hist.setdefault(name, {"K": cols.shape[1], "h": np.zeros(256)})
        p = (cols[:, :, None, :] * w.T[None, :, :, None]) & 255
        h["h"] += np.bincount(p.flatten().numpy(), minlength=256)

    orig = qnet.conv

    def conv(name, xin, acc, calibrate, stats):
        L = qnet.layers[name]
        xq, _ = qnet._qa(name, xin, False)
        cols = F.unfold(xq.double(), 3, padding=1, stride=L.get("stride", 1)).long()
        collect(name, cols[:8], L["wq"].flatten(1))
        return orig(name, xin, None, False, None)
    qnet.conv = conv
    try:
        for i in range(0, len(x), 100):
            y = qnet.forward(x[i:i + 100], None)
    finally:
        qnet.conv = orig
    return hist


def calibrated_preloads(design, hist):
    """Preload -round(E[E_K]) per layer from Theorem 3 under the measured residue law."""
    if design in ("exact", "int8_exact_ref"):
        return {}
    k, z, c = low_tables(design)
    q = 1 << k
    pre = {}
    for name, h in hist.items():
        pmf = h["h"].reshape(-1, q).sum(0) if q <= 256 else None
        mean, _ = moments(z, c, [h["K"]], pmf=pmf)[h["K"]]
        pre[name] = int(-round(mean))
    return pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--compensate", choices=["none", "uniform", "calibrated"], default="none")
    ap.add_argument("--out", default="cnn_results.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ACC_THREADS", 20)))
    model = ResNet20(classes=CLASSES)
    model.load_state_dict(torch.load(CKPT, weights_only=True)["state_dict"])
    model.eval()
    xtr, _ = load_cifar(True)
    xte, yte = load_cifar(False)
    g = torch.Generator().manual_seed(1)
    calib = xtr[torch.randperm(len(xtr), generator=g)[:500]]
    with torch.no_grad():
        qnet = QNet(model, calib)
        xs, ys = xte[:a.n], yte[:a.n]
        float_acc = (model(xs).argmax(1) == ys).float().mean().item()
        hist = residue_histograms(qnet, calib) if a.compensate == "calibrated" else None
        for d in a.designs:
            t0 = time.time()
            pre = calibrated_preloads(d, hist) if a.compensate == "calibrated" else None
            acc = None if d == "int8_exact_ref" else Accumulator(d, a.compensate, pre)
            preds, stats = [], {}
            for i in range(0, a.n, a.bs):
                xb = xs[i:i + a.bs]
                if acc is not None:
                    acc._K = None
                preds.append(qnet_forward(qnet, xb, acc, stats).argmax(1))
            preds = torch.cat(preds)
            row = {"design": d, "compensate": a.compensate, "n": a.n, "top1": (preds == ys).float().mean().item(),
                   "float_top1": float_acc, "preds": preds.tolist(), "sec": time.time() - t0,
                   "layer_err": {k: {"K": v["K"], "mean": v["s1"] / v["n"], "mse": v["s2"] / v["n"],
                                     "hist256": v["hist256"].astype(int).tolist()}
                                 for k, v in stats.items()}}
            with open(Path(__file__).with_name(a.out), "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(d, a.compensate, round(row["top1"], 4), round(row["sec"]), flush=True)


def qnet_forward(qnet, xb, acc, stats):
    if acc is None:
        return qnet.forward(xb, None)
    # the compensation constant depends on K, which is fixed per layer; set it lazily
    orig = acc.run

    def run(prods):
        prods = list(prods)
        acc._K = len(prods)
        return orig(iter(prods))
    acc.run = run
    try:
        return qnet.forward(xb, acc, stats=stats)
    finally:
        acc.run = orig


if __name__ == "__main__":
    main()
