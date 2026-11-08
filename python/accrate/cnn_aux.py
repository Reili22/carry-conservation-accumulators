"""Overflow and requantization analysis of the INT8 ResNet-20 accumulators (exact arithmetic).

Per layer: K, the largest |exact accumulator| over the evaluated images
(checked against the signed 24-bit range), and the number of accumulator LSBs
below the next quantization step, log2(s_next / (s_a * s_w)) per output channel.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cnn_accumulate import QNet
from resnet20 import CKPT, ResNet20, load_cifar

HERE = Path(__file__).parent


def main(n=1000):
    torch.set_num_threads(8)
    model = ResNet20(); model.load_state_dict(torch.load(CKPT, weights_only=True)["state_dict"]); model.eval()
    xtr, _ = load_cifar(True); xte, _ = load_cifar(False)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        q = QNet(model, xtr[torch.randperm(len(xtr), generator=g)[:500]])
        maxabs = {}
        orig = q.conv

        def conv(name, x, acc, calibrate, stats):
            L = q.layers[name]
            xq, sa = q._qa(name, x, False)
            cols = F.unfold(xq.double(), 3, padding=1, stride=L.get("stride", 1))
            out = torch.einsum("bkl,ck->bcl", cols, L["wq"].flatten(1).double())
            maxabs[name] = max(maxabs.get(name, 0.0), float(out.abs().max()))
            return orig(name, x, None, False, None)
        q.conv = conv
        for i in range(0, n, 100):
            q.forward(xte[i:i + 100], None)
    names = list(q.layers)
    nxt = {names[i]: names[i + 1] for i in range(len(names) - 1)}
    rows = []
    for name, L in q.layers.items():
        if name == "fc":
            continue
        K = L["wq"].flatten(1).shape[1]
        s_next = q.act_scale[nxt[name]]
        lsb = q.act_scale[name] * L["sw"].double().numpy()
        drop = np.log2(s_next / lsb)
        rows.append({"layer": name, "K": K, "max_abs_acc": maxabs[name], "fits_24bit": maxabs[name] < 2 ** 23,
                     "bits_below_next_lsb_median": float(np.median(drop)), "bits_below_next_lsb_min": float(drop.min())})
    (HERE / "cnn_aux.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)
    print("max |acc| overall:", max(r["max_abs_acc"] for r in rows), "limit", 2 ** 23)


if __name__ == "__main__":
    main()
