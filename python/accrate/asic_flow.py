"""Synthesize accumulators to Nangate45, verify netlists bit-exactly, estimate energy.

Flow (identical for every design):
  Yosys 0.70  read_verilog; synth -flatten -noabc; write_blif   (generic gates + latches)
  ABC         read_lib (X1 subset); strash; dc2; &dch -f; &nf; stime -p; write_blif
No retiming or sequential optimization is run in ABC, so registers are those
left by Yosys (constant registers are removed by Yosys opt_dff).  Area is the
ABC-mapped gate area plus one DFF_X1 per latch; delay is ABC's `stime`
register-to-register/IO combinational critical path.  The mapped BLIF is then
simulated cycle by cycle (zero delay) on an operand trace and compared with the
Python reference model at every cycle; energy uses liberty.py's model.
(The Windows Yosys build mis-reads ABC results back, so ABC is run standalone.)
"""
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from acc_designs import IW, MASK, SIGNED, operand24, rtl, step_fn  # noqa: E402
import netsim  # noqa: E402
from liberty import py_function, read  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / ("asic_s" if SIGNED else "asic")
LIB = Path(__file__).parent / "asic" / "nangate45_x1.lib"  # X1-drive subset of NangateOpenCellLibrary_typical
_OSS = Path(r"D:/tools/oss-cad-suite")  # oss-cad-suite 2026-10-21 (Yosys 0.70+186)
YOSYS = str(_OSS / "bin/yosys.exe") if (_OSS / "bin/yosys.exe").exists() else shutil.which("yosys")
ABC = str(_OSS / "bin/yosys-abc.exe") if (_OSS / "bin/yosys-abc.exe").exists() else shutil.which("yosys-abc")
FF_CELL = "DFF_X1"


def _env():
    b = Path(YOSYS).parent
    return dict(os.environ, PATH=os.pathsep.join([str(b), str(b.parent / "lib"), os.environ.get("PATH", "")]))


def synth(design):
    OUT.mkdir(exist_ok=True)
    if not (OUT / LIB.name).exists():
        shutil.copy(LIB, OUT / LIB.name)
    top, text, extra = rtl(design)
    src = OUT / f"{top}.v"
    src.write_text(text)
    (OUT / "abc.constr").write_text("set_driving_cell BUF_X1\nset_load 3\n")
    gen, mapped = OUT / f"{top}_gen.blif", OUT / f"{top}_map.blif"
    files = " ".join(f'"{p}"' for p in [src, *extra])
    for _attempt in range(3):  # the Windows build crashes intermittently; outputs are re-verified
        ys = subprocess.run([YOSYS, "-Q", "-T", "-p",
                             f"read_verilog {files}; synth -top {top} -flatten -noabc; dffunmap; opt_clean; "
                             f"write_blif -impltf \"{gen}\""],
                            capture_output=True, text=True, env=_env())
        if ys.returncode == 0:
            break
    if ys.returncode:
        raise RuntimeError(f"{design}: yosys failed\n{ys.stdout[-1500:]}{ys.stderr[-1500:]}")
    lib = read(str(LIB))
    log_text = ys.stdout
    maps = []
    for tag, opt in ABC_SCRIPTS.items():
        mapped = OUT / f"{top}_map_{tag}.blif"
        script = OUT / f"{top}_{tag}.abc"
        # ABC cannot parse paths with spaces, so it runs inside OUT with bare file names
        script.write_text(f"read_lib -w {LIB.name}\nread_blif {gen.name}\nread_constr -v abc.constr\n"
                          f"{opt}\nstime -p\nwrite_blif {mapped.name}\n")
        ab = subprocess.run([ABC, "-f", script.name], cwd=OUT, capture_output=True, text=True, env=_env())
        log_text += ab.stdout + ab.stderr
        m = re.search(r"Area =\s*([0-9.]+).*?Delay =\s*([0-9.]+) ps", ab.stdout)
        if ab.returncode or m is None or not mapped.exists():
            raise RuntimeError(f"{design}/{tag}: abc failed\n{ab.stdout[-1500:]}")
        net = Netlist(mapped, lib)
        area = (float(m.group(1)) - net.port_buffers * lib["cells"]["BUF_X1"]["area"]
                + len(net.latches) * lib["cells"][FF_CELL]["area"])
        maps.append({"tag": tag, "area_um2": area, "delay_ps": float(m.group(2)), "netlist": str(mapped),
                     "n_ff": len(net.latches),
                     "logic_area_um2": area - len(net.latches) * lib["cells"][FF_CELL]["area"]})
    log = OUT / f"{top}.log"
    log.write_text(log_text)
    return {"design": design, "top": top, "maps": {r.pop("tag"): r for r in maps},
            "log_sha256": sha256(log.read_bytes()).hexdigest()}


# Three structural pre-optimizations, each followed by a sweep of &nf's delay-relaxation ratio R: R=0 is the
# fastest mapping and large R approaches minimum area.  Each design thus gets an
# area/delay curve; iso-delay comparisons read the cheapest point meeting a target.
# resyn2 is spelled out because abc.rc is not loaded.
_RESYN2 = "balance\nrewrite\nrefactor\nbalance\nrewrite\nrewrite -z\nbalance\nrefactor -z\nrewrite -z\nbalance"
_BASES = {"rs": f"strash\n{_RESYN2}\n&get -n\n&dch -f\n",
          "dc": "strash\ndc2\n&get -n\n&dch -f\n",
          "pl": "strash\n&get -n\n&dch -f\n"}
ABC_SCRIPTS = {f"{b}R{r}": s + f"&nf -p -R {r}\n&put"
               for b, s in _BASES.items() for r in (0, 10, 25, 50, 100, 1000)}


class Netlist:
    """Mapped BLIF (.gate/.latch/.names) with cycle-based simulation and energy accounting."""

    def __init__(self, path, lib):
        self.lib = lib
        text = Path(path).read_text().replace("\\\n", " ")
        self.inputs, self.outputs, self.gates, self.latches, self.names = [], [], [], [], []
        self.port_buffers = 0
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            tok = lines[i].split()
            i += 1
            if not tok:
                continue
            if tok[0] == ".inputs":
                self.inputs += tok[1:]
            elif tok[0] == ".outputs":
                self.outputs += tok[1:]
            elif tok[0] == ".gate":
                typ = tok[1]
                conn = dict(t.split("=", 1) for t in tok[2:])
                if typ in ("_const0_", "_const1_"):  # ABC pseudo-gates for constant outputs
                    self.names.append(([], next(iter(conn.values())), [["1"]] if typ == "_const1_" else []))
                    continue
                if typ == "BUF_X1" and conn.get("Z", "").startswith("out["):
                    # port buffer ABC inserts for register-driven outputs: modeled as a wire
                    self.names.append(([conn["A"]], conn["Z"], [["1", "1"]]))
                    self.port_buffers += 1
                    continue
                pins = lib["cells"][typ]["pins"]
                ins = {p: n for p, n in conn.items() if pins[p]["dir"] == "input"}
                outs = {p: (n, compile(py_function(pins[p]["func"]), typ, "eval"))
                        for p, n in conn.items() if pins[p]["dir"] == "output"}
                self.gates.append((typ, ins, outs))
            elif tok[0] == ".latch":
                self.latches.append((tok[1], tok[2], int(tok[3]) if len(tok) > 3 and tok[3] in "01" else 0))
            elif tok[0] == ".names":
                sig = tok[1:]
                rows = []
                while i < len(lines) and lines[i] and not lines[i].startswith("."):
                    rows.append(lines[i].split()); i += 1
                self.names.append((sig[:-1], sig[-1], rows))
        self._order()
        self._loads()

    def _order(self):
        drv = {}
        for idx, (_, _, outs) in enumerate(self.gates):
            for n, _c in outs.values():
                drv[n] = ("g", idx)
        for idx, (ins, o, _r) in enumerate(self.names):
            drv[o] = ("n", idx)
        order, state = [], {}

        def deps(node):
            kind, idx = node
            return self.gates[idx][1].values() if kind == "g" else self.names[idx][0]

        def visit(node):
            if state.get(node) == 2:
                return
            assert state.get(node) != 1, "combinational loop"
            state[node] = 1
            for n in deps(node):
                if n in drv:
                    visit(drv[n])
            state[node] = 2
            order.append(node)
        for node in set(drv.values()):
            visit(node)
        self.order = order

    def _loads(self):
        self.load = {}
        for typ, ins, _ in self.gates:
            for p, n in ins.items():
                self.load[n] = self.load.get(n, 0.0) + self.lib["cells"][typ]["pins"][p]["cap"]
        dcap = self.lib["cells"][FF_CELL]["pins"]["D"]["cap"]
        for d, _q, _i in self.latches:
            self.load[d] = self.load.get(d, 0.0) + dcap

    def _eval_names(self, idx, val):
        ins, o, rows = self.names[idx]
        if not ins:
            val[o] = bool(rows and rows[0] == ["1"])
            return
        hit = False
        for r in rows:
            pat, outv = (r[0], r[1]) if len(r) == 2 else ("", r[0])
            if all(c == "-" or (c == "1") == val[n] for c, n in zip(pat, ins)):
                hit = outv == "1"; break
        val[o] = hit

    def simulate(self, U, ref_step):
        lib, V = self.lib, self.lib["V"]
        ff = lib["cells"][FF_CELL]
        e_ck = ff["pins"]["CK"]["e_int"] * 2 + ff["pins"]["CK"]["cap"] * V * V
        e_q = ff["pins"]["Q"]["e_int"]
        e_d = ff["pins"]["D"]["e_int"]
        in_bits = [n for n in self.inputs if n.startswith("in[")]
        out_bits = sorted((n for n in self.outputs if n.startswith("out[")), key=lambda s: int(s[4:-1]))
        state = {q: bool(init) for _d, q, init in self.latches}
        val, prev = {}, None
        e_sw = e_int = 0.0
        ok, T_ref = True, None
        for t in range(len(U)):
            val.update(state)
            for n in in_bits:
                val[n] = bool((int(U[t]) >> int(n[3:-1])) & 1)
            if "clr" in self.inputs:
                val["clr"] = t == 0  # clear on the first cycle, accumulate afterwards
            for kind, idx in self.order:
                if kind == "n":
                    self._eval_names(idx, val)
                    continue
                typ, ins, outs = self.gates[idx]
                env = {p: np.bool_(val[n]) for p, n in ins.items()}
                for p, (n, code) in outs.items():
                    val[n] = bool(eval(code, {}, env))
            out = sum(1 << int(n[4:-1]) for n in out_bits if val.get(n, False))
            if T_ref is not None and out != T_ref:
                ok = False
            T_ref = 0 if t == 0 else int(ref_step(np.array([T_ref]), np.array([int(U[t])]))[0]) & MASK
            if prev is not None:
                for kind, idx in self.order:
                    if kind != "g":
                        continue
                    typ, ins, outs = self.gates[idx]
                    pins = lib["cells"][typ]["pins"]
                    for p, (n, _c) in outs.items():
                        if val[n] != prev[n]:
                            e_sw += 0.5 * self.load.get(n, 0.0) * V * V
                            e_int += pins[p]["e_int"]
                    for p, n in ins.items():
                        if val[n] != prev[n]:
                            e_int += pins[p]["e_int"]
                for n in in_bits:
                    if val[n] != prev[n]:
                        e_sw += 0.5 * self.load.get(n, 0.0) * V * V
                for d, q, _i in self.latches:
                    if val[d] != prev[d]:
                        e_int += e_d
            prev = dict(val)
            new = {}
            for d, q, _i in self.latches:
                new[q] = val[d]
                if new[q] != state[q]:
                    e_sw += 0.5 * self.load.get(q, 0.0) * V * V
                    e_int += e_q
                e_int += e_ck
            state = new
        n = max(len(U) - 1, 1)
        leak = sum(lib["cells"][g[0]]["leak_nW"] for g in self.gates) + len(self.latches) * ff["leak_nW"]
        return ok, {"e_dyn_fJ": (e_sw + e_int) / n, "e_sw_fJ": e_sw / n, "e_int_fJ": e_int / n,
                    "leak_nW": leak, "n_ff": len(self.latches), "n_gates": len(self.gates)}


def trace(n=3000, seed=11):
    """MAC-like operand trace: products of independent uniform unsigned 8-bit values."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, n) * rng.integers(0, 256, n)) & ((1 << IW) - 1)


def cec(gen, mapped):
    """Formal combinational equivalence (registers matched by name) of a mapped netlist and its RTL elaboration."""
    script = OUT / f"{Path(mapped).stem}_cec.abc"
    script.write_text(f"read_lib -w {LIB.name}\ncec {Path(gen).name} {Path(mapped).name}\n")
    r = subprocess.run([ABC, "-f", script.name], cwd=OUT, capture_output=True, text=True, env=_env())
    return "Networks are equivalent" in r.stdout


def evaluate(design, K=144):
    row = synth(design)
    lib = read(str(LIB))
    U16, clr = netsim.mac_trace(K=K, signed=SIGNED)
    T = netsim.reference(step_fn(design), operand24(U16), clr)
    gen = OUT / f"acc_{design}_gen.blif"
    for tag, m in row["maps"].items():
        ok, energy = netsim.run(netsim.parse_blif(m["netlist"], lib), lib, U16, clr, T)
        m.update(energy)
        m["e_dyn_fJ"] = energy["e_unit_fJ"]  # glitch-aware energy is the reported energy
        m["matches_reference"] = ok
        m["cec_equivalent"] = cec(gen, m["netlist"])
    row["all_match"] = all(m["matches_reference"] for m in row["maps"].values())
    row["all_cec"] = all(m["cec_equivalent"] for m in row["maps"].values())
    row["signed"] = SIGNED
    return row


def main(designs, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(evaluate, designs))


if __name__ == "__main__":
    for r in main(sys.argv[1:] or ["exact", "round_4", "evo_add8u_6SM"]):
        print(r["design"], r["all_match"], {t: (round(m["area_um2"], 1), round(m["delay_ps"]), round(m["e_dyn_fJ"]))
                                           for t, m in r["maps"].items()})
