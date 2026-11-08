"""Netlist-level proof of dominance for state-blind lower-part accumulators.

For a state-blind design (carry c depends only on the operand residue), the
low-part registers T[0..k-1] cannot influence T[k..].  Removing every gate
outside the fan-in cone of the upper registers therefore yields, from the
design's *own* mapped netlist, an accumulator for the quantized operand
Q(U) = U - r + q*g(r).  The pruned netlist is re-simulated bit-exactly against
that quantizer and re-timed with ABC; it has no more gates or registers than
the original, hence no more area and no larger delay or switching energy.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from acc_designs import MASK, SIGNED, operand24  # noqa: E402
from asic_flow import ABC, FF_CELL, LIB, OUT, _env, cec  # noqa: E402
import netsim  # noqa: E402
from liberty import read  # noqa: E402
from theory import low_tables  # noqa: E402

TAG = "s" if SIGNED else "u"
ROWS = json.loads(Path(__file__).with_name(f"hw_study_{TAG}.json").read_text())


def twin_step(design):
    k, _z, c = low_tables(design)
    q = 1 << k
    g = c[0].astype(np.int64)

    def f(T, U):
        return ((((T >> k) + (U >> k) + g[U & (q - 1)]) << k)) & MASK
    return k, f


def prune(blif_path, k, out_path):
    text = Path(blif_path).read_text().replace("\\\n", " ")
    lines = text.splitlines()
    gates, latches, names, header = [], [], [], []
    i = 0
    while i < len(lines):
        ln = lines[i]; tok = ln.split(); i += 1
        if not tok:
            continue
        if tok[0] == ".gate":
            conn = dict(t.split("=", 1) for t in tok[2:])
            gates.append((ln, tok[1], conn))
        elif tok[0] == ".latch":
            latches.append((ln, tok[1], tok[2]))
        elif tok[0] == ".names":
            body = [ln]
            while i < len(lines) and lines[i] and not lines[i].startswith("."):
                body.append(lines[i]); i += 1
            names.append((body, tok[1:-1], tok[-1]))
        elif tok[0] in (".model", ".inputs"):
            header.append(ln)
    bit = lambda n: int(re.search(r"\[(\d+)\]", n).group(1))  # noqa: E731
    lib = read(str(LIB))
    keep_latch = [lt for lt in latches if lt[2].startswith("T[") and bit(lt[2]) >= k]
    keep_outs = [f"out[{j}]" for j in range(k, 64)]
    driver = {}
    for g in gates:
        pins = lib["cells"].get(g[1], {"pins": {}})["pins"]
        for p, n in g[2].items():
            if p in pins and pins[p]["dir"] == "output" or g[1].startswith("_const"):
                driver[n] = ("g", g)
    for nm in names:
        driver[nm[2]] = ("n", nm)
    low_q = {lt[2] for lt in latches if lt not in keep_latch}
    need = [lt[1] for lt in keep_latch] + [o for o in keep_outs if o in driver]
    kept_g, kept_n, seen, touched_low = [], [], set(), False
    while need:
        n = need.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in low_q:
            touched_low = True
            continue
        d = driver.get(n)
        if d is None:
            continue
        if d[0] == "g":
            g = d[1]
            if g not in kept_g:
                kept_g.append(g)
                pins = lib["cells"].get(g[1], {"pins": {}})["pins"]
                need += [v for p, v in g[2].items() if p in pins and pins[p]["dir"] == "input"]
        else:
            if d[1] not in kept_n:
                kept_n.append(d[1]); need += list(d[1][1])
    outs = [o for o in keep_outs if o in driver]
    core = [g[0] for g in kept_g] + [lt[0] for lt in keep_latch]
    for nm in kept_n:
        core += nm[0]
    # timing view: mapped cells only
    Path(out_path).with_name(Path(out_path).stem + "_t.blif").write_text(
        "\n".join(header + [".outputs " + " ".join(outs)] + core) + "\n.end\n")
    # interface view: the twin's low outputs are constant zero (for equivalence checking)
    low_outs = [f"out[{j}]" for j in range(k)]
    Path(out_path).write_text("\n".join(header + [".outputs " + " ".join(low_outs + outs)] + core
                                        + [f".gate _const0_ z={o}" for o in low_outs]) + "\n.end\n")
    return touched_low, len(kept_g), len(gates), len(keep_latch), len(latches)


def abc_delay(blif):
    s = OUT / "prune_stime.abc"
    s.write_text(f"read_lib -w {LIB.name}\nread_constr -v abc.constr\nread {Path(blif).name}\ntopo\nstime -p\n")
    r = subprocess.run([ABC, "-f", s.name], cwd=OUT, capture_output=True, text=True, env=_env())
    m = re.search(r"Area =\s*([0-9.]+).*?Delay =\s*([0-9.]+) ps", r.stdout)
    if m is None:
        raise RuntimeError(f"stime failed on {blif}\n{r.stdout[-800:]}")
    return float(m.group(2))


def fill_delays():
    """Re-time already pruned netlists (used after changing the timing script only)."""
    path = Path(__file__).with_name(f"prune_check_{TAG}.json")
    rep = json.loads(path.read_text())
    for x in rep:
        x["delay"] = abc_delay(OUT / f"acc_{x['design']}_pruned_{x['map']}.blif")
    path.write_text(json.dumps(rep, indent=1))
    summarize(rep)


def summarize(report):
    ok = [x for x in report if x["twin_matches"] and not x["touched_low_state"]]
    # tolerances: area summation rounding (0.01 um^2), ABC timing print resolution (0.5 ps)
    worse = [x for x in ok if x["area"] > x["area_orig"] + 0.01 or x["delay"] > x["delay_orig"] + 0.5
             or x["energy"] > x["energy_orig"] + 1e-6]
    print(f"{len(report)} mapped netlists of {len({x['design'] for x in report})} state-blind designs; "
          f"pruned twin verified: {len(ok)}; twin worse on some axis: {len(worse)}")
    for x in worse[:10]:
        print("  WORSE", x["design"], x["map"], x["area"], x["area_orig"], x["delay"], x["delay_orig"],
              x["energy"], x["energy_orig"])
    for key, orig in (("area", "area_orig"), ("energy", "energy_orig"), ("delay", "delay_orig")):
        s = [1 - x[key] / x[orig] for x in ok]
        print(f"  {key}: median saving {100 * np.median(s):.1f}%, min {100 * min(s):.1f}%, max {100 * max(s):.1f}%")


def twin_rtl(design, k):
    """Name of the round/trunc/tplus design equal to the quantizer twin, if there is one."""
    _k, _z, c = low_tables(design)
    g = c[0]
    q = 1 << k
    if not g.any():
        return f"trunc_{k}"
    if np.all(g == 1):
        return f"tplus_{k}"
    if np.array_equal(g, (np.arange(q) >> (k - 1)) & 1):
        return f"round_{k}"
    return None


def main():
    lib = read(str(LIB))
    report = []
    U16, clr = netsim.mac_trace(signed=SIGNED)
    for r in ROWS:
        d = r["design"]
        if d.startswith(("exact", "round_", "trunc_", "tplus_")) or not r["state_blind"] or r["k"] == 0:
            continue
        k, step = twin_step(d)
        T = netsim.reference(step, operand24(U16), clr)
        twin = twin_rtl(d, k)
        for tag, m in r["hw"].items():
            src = OUT / f"acc_{d}_map_{tag}.blif"
            dst = OUT / f"acc_{d}_pruned_{tag}.blif"
            touched, ng, ng0, nl, nl0 = prune(src, k, dst)
            ok, en = netsim.run(netsim.parse_blif(dst, lib), lib, U16, clr, T)
            area = (sum(lib["cells"][g[1]]["area"] for g in _gates(dst)) + nl * lib["cells"][FF_CELL]["area"])
            formal = cec(OUT / f"acc_{twin}_gen.blif", dst) if twin else None
            report.append({"design": d, "map": tag, "twin": twin, "touched_low_state": touched, "twin_matches": ok,
                           "twin_cec": formal, "area": area, "area_orig": m["area_um2"],
                           "delay": abc_delay(dst.with_name(dst.stem + "_t.blif")), "delay_orig": m["delay_ps"],
                           "energy": en["e_unit_fJ"], "energy_orig": m["e_unit_fJ"],
                           "gates": ng, "gates_orig": ng0, "ff": nl, "ff_orig": nl0})
    Path(__file__).with_name(f"prune_check_{TAG}.json").write_text(json.dumps(report, indent=1))
    summarize(report)
    print("formal twin equivalence:", sum(x["twin_cec"] is True for x in report), "proved,",
          sum(x["twin_cec"] is False for x in report), "failed,", sum(x["twin_cec"] is None for x in report),
          "without a round/trunc twin RTL")


def _gates(blif):
    out = []
    for ln in Path(blif).read_text().replace("\\\n", " ").splitlines():
        t = ln.split()
        if t and t[0] == ".gate" and not t[1].startswith("_const") and not (
                t[1] == "BUF_X1" and any(x.startswith("Z=out[") for x in t[2:])):
            out.append(t)
    return out


if __name__ == "__main__":
    fill_delays() if sys.argv[1:] == ["--delays"] else main()
