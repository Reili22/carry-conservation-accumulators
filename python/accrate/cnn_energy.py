"""Energy of every signed accumulator netlist on real ResNet-20 operand streams.

The trace concatenates the serial product sequences of randomly chosen output
positions of two layers (K = 144 and K = 576) on test images, with a clear at
the start of each accumulation, i.e. what a MAC unit actually sees.  Every
mapped netlist of hw_study_s.json is re-simulated (functional check included)
and its unit-delay energy per operation recorded as e_cnn_fJ.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import os  # noqa: E402
os.environ["ACC_SIGNED"] = "1"
from acc_designs import operand24, step_fn  # noqa: E402
from asic_flow import LIB, OUT  # noqa: E402
from cnn_accumulate import QNet  # noqa: E402
from liberty import read  # noqa: E402
import netsim  # noqa: E402
from resnet20 import CKPT, ResNet20, load_cifar  # noqa: E402

HERE = Path(__file__).parent


def cnn_trace(layers=("l2a", "l7b"), per_layer=14, seed=3):
    torch.set_num_threads(8)
    model = ResNet20(); model.load_state_dict(torch.load(CKPT, weights_only=True)["state_dict"]); model.eval()
    xtr, _ = load_cifar(True); xte, _ = load_cifar(False)
    g = torch.Generator().manual_seed(1)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        q = QNet(model, xtr[torch.randperm(len(xtr), generator=g)[:500]])
        seqs = []
        orig = q.conv

        def conv(name, x, acc, calibrate, stats):
            if name in layers:
                L = q.layers[name]
                xq, _ = q._qa(name, x, False)
                cols = F.unfold(xq.double(), 3, padding=1, stride=L.get("stride", 1)).long()
                w = L["wq"].flatten(1)
                for _ in range(per_layer):
                    b, l, c = rng.integers(cols.shape[0]), rng.integers(cols.shape[2]), rng.integers(w.shape[0])
                    seqs.append((cols[b, :, l] * w[c]).numpy())
            return orig(name, x, None, False, None)
        q.conv = conv
        q.forward(xte[:4], None)
    U = np.concatenate(seqs) & 0xFFFF
    clr = np.concatenate([np.arange(len(s)) == 0 for s in seqs])
    return U.astype(np.int64), clr


def main():
    U16, clr = cnn_trace()
    lib = read(str(LIB))
    rows = json.loads((HERE / "hw_study_s.json").read_text())
    bad = []
    for r in rows:
        T = netsim.reference(step_fn(r["design"]), operand24(U16), clr)
        for tag, m in r["hw"].items():
            path = OUT / f"acc_{r['design']}_map_{tag}.blif"
            ok, en = netsim.run(netsim.parse_blif(path, lib), lib, U16, clr, T)
            m["e_cnn_fJ"] = en["e_unit_fJ"]
            if not ok:
                bad.append((r["design"], tag))
    (HERE / "hw_study_s.json").write_text(json.dumps(rows, indent=1))
    print(f"CNN trace: {len(U16)} operations, {int(clr.sum())} accumulations; mismatches: {bad}")


if __name__ == "__main__":
    main()
