"""Exploratory (post hoc, after the pre-registered H2 failed): why does Theorem 3 under one pooled
residue law underpredict the accumulated-error variance of truncation-type designs?

For each conv layer and 24 test images, the twin error of trunc_k / round_k is computed exactly from
the products and compared with (a) Theorem 3 under the pooled law, (b) the per-output-channel laws,
and (c) Theorem 3 conditioned on the number N of nonzero products per output,
Var = E[N] var_nz + Var(N) mean_nz^2.  Writes cnn_sparsity_diag_<dataset>.json.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from cnn_accumulate import QNet
from resnet20 import CKPT, CLASSES, ResNet20, load_cifar
from theory import low_tables, moments
torch.set_num_threads(8)
m = ResNet20(classes=CLASSES); m.load_state_dict(torch.load(CKPT, weights_only=True)["state_dict"]); m.eval()
xtr, _ = load_cifar(True); xte, _ = load_cifar(False)
g = torch.Generator().manual_seed(1)
qn = QNet(m, xtr[torch.randperm(len(xtr), generator=g)[:500]])
designs = sys.argv[1:] or ["trunc_4", "trunc_6", "round_5"]
tabs = {d: low_tables(d) for d in designs}
rec = {}
orig = QNet.conv
def conv(self, name, x, acc, calibrate, stats):
    if not calibrate:
        L = self.layers[name]; xq, _ = self._qa(name, x, False)
        cols = F.unfold(xq.double(), 3, padding=1, stride=L.get("stride", 1)).long()
        w = L["wq"].flatten(1)
        p = (cols[:, :, None, :] * w.T[None, :, :, None])  # B,K,C,L
        analyse(name, p); del p
    return orig(self, name, x, acc, calibrate, stats)
out = {}
def analyse(name, p):
    K = p.shape[1]
    for d in designs:
        k, z, c = tabs[d]; q = 1 << k
        if not (d.startswith("trunc") or d.startswith("round")): continue
        r = (p & (q - 1)).double()
        e = (-r if d.startswith("trunc") else torch.where(r >= q // 2, q - r, -r)).sum(1)  # B,C,L exact-twin error
        meas = float(e.var())
        pooled = np.bincount(r.long().flatten().numpy(), minlength=q).astype(float)
        mu, v = moments(z, c, [K], pmf=pooled)[K]
        # channel-conditional: total var = E_c[var_c] + Var_c[mean_c]
        mc, vc = [], []
        for ch in range(p.shape[2]):
            h = np.bincount(r[:, :, ch].long().flatten().numpy(), minlength=q).astype(float)
            a, b = moments(z, c, [K], pmf=h)[K]; mc.append(a); vc.append(b)
        chan = float(np.mean(vc) + np.var(mc))
        # condition on the number of nonzero products per output (ReLU sparsity)
        nz = (p != 0)
        n = nz.sum(1).double()
        hnz = np.bincount(r[nz].long().numpy(), minlength=q).astype(float)
        hnz /= hnz.sum(); vals = np.arange(q, dtype=float)
        xv = (-vals if d.startswith("trunc") else np.where(vals >= q // 2, q - vals, -vals))
        mnz = float((hnz * xv).sum()); vnz = float((hnz * xv * xv).sum()) - mnz ** 2
        spars = float(n.mean()) * vnz + float(n.var()) * mnz ** 2
        out.setdefault(d + "_sparsity", []).append((name, spars / meas, spars / meas))
        out.setdefault(d, []).append((name, v / meas, chan / meas))
QNet.conv = conv
with torch.no_grad():
    qn.forward(xte[:24], None)
for d, rows in out.items():
    a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
    print(f"{d}: pooled-law var ratio median {np.median(a):.3f} [{a.min():.3f},{a.max():.3f}]; "
          f"per-channel-law {np.median(b):.3f} [{b.min():.3f},{b.max():.3f}]")
json.dump(out, open(f"cnn_sparsity_diag_{os.environ.get('ACC_DATASET', 'cifar10')}.json", "w"), indent=1)
