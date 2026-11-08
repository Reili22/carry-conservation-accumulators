"""Diffusion versus effective state count against the rounding bound ((q/S)^2-1)/12.

S is the number of recurrent low-part states reached from reset (X_0 = 0) under
iid uniform residues; every cell must satisfy diffusion >= ((q/S)^2-1)/12.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from carry_census import rates, tables  # noqa: E402
from evo_c2py import EVO_ROOT, cells  # noqa: E402


def recurrent_states(z):
    q = z.shape[0]
    reach = {0}
    frontier = [0]
    while frontier:
        x = frontier.pop()
        for y in set(z[x].tolist()):
            if y not in reach:
                reach.add(y); frontier.append(y)
    # recurrent = states in a closed communicating class reachable from 0
    succ = {x: set(z[x].tolist()) for x in reach}
    rec = set()
    for x in reach:
        seen, stack = {x}, [x]
        while stack:
            for y in succ[stack.pop()]:
                if y not in seen:
                    seen.add(y); stack.append(y)
        if all(x in _reach(succ, y) for y in seen):
            rec.add(x)
    return rec


def _reach(succ, s):
    seen, stack = {s}, [s]
    while stack:
        for y in succ[stack.pop()]:
            if y not in seen:
                seen.add(y); stack.append(y)
    return seen


def main():
    census = {r["cell"]: r for r in json.loads(Path(__file__).with_name("carry_census.json").read_text())}
    rows = []
    for width in (8, 12, 16):
        for name, f, _ in cells(EVO_ROOT / f"{width}_unsigned"):
            r = census.get(name)
            if not r or not r.get("k"):
                continue
            k = r["k"]; q = 1 << k
            for side, tag in ((True, "L"), (False, "R")):
                z, c = tables(f, k, side)
                rec = recurrent_states(z)
                S = len(rec)
                bound = ((q / S) ** 2 - 1) / 12
                drift, diff = r[f"drift_{tag}"], r[f"diff_{tag}"]
                rows.append(dict(cell=name, w=width, k=k, side=tag, S=S, diffusion=diff, drift=drift,
                                 bound=bound, slack=diff - bound, stateless=r[f"stateless_{tag}"],
                                 t_eq=float(np.log2(12 * diff + 1) / 2)))
    viol = [x for x in rows if x["slack"] < -1e-6]
    tight = [x for x in rows if abs(x["slack"]) < 1e-6]
    print(f"{len(rows)} cell-orientations, violations={len(viol)}, tight={len(tight)}")
    for x in rows:
        if x["side"] == "L":
            print(f"{x['cell']:12s} k={x['k']} S={x['S']:4d} log2S={np.log2(x['S']):5.2f} diff={x['diffusion']:9.3f} "
                  f"bound={x['bound']:9.3f} t_eq={x['t_eq']:5.2f} {'blind' if x['stateless'] else 'aware'}")
    Path(__file__).with_name("state_bound.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
