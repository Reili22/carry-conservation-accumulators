"""Vectorized simulation of a mapped accumulator netlist over a whole operand trace.

The register state of every cycle is known from the reference model, so all
cycles are evaluated in parallel.  The simulator
  * checks that every register's next-state net equals the reference next state
    on every cycle (functional verification of every bit, not only outputs);
  * counts zero-delay toggles (settled value changes between cycles); and
  * counts unit-delay toggles: starting from the previous settled values, all
    gates update simultaneously with unit delay until the network is stable,
    which includes glitches.
Energy per toggle follows liberty.py (internal energy of the switching pin plus
0.5*C*V^2 on the driven capacitance); every register adds its clock energy per
cycle.  Liberty energy unit: fJ.
"""
import numpy as np

from liberty import py_function

MASK24 = (1 << 24) - 1


def parse_blif(path, lib):
    text = open(path).read().replace("\\\n", " ")
    inputs, outputs, gates, latches, wires = [], [], [], [], []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        tok = lines[i].split(); i += 1
        if not tok:
            continue
        if tok[0] == ".inputs":
            inputs += tok[1:]
        elif tok[0] == ".outputs":
            outputs += tok[1:]
        elif tok[0] == ".latch":
            latches.append((tok[1], tok[2]))
        elif tok[0] == ".gate":
            typ, conn = tok[1], dict(t.split("=", 1) for t in tok[2:])
            if typ in ("_const0_", "_const1_"):
                wires.append((None, next(iter(conn.values())), typ == "_const1_"))
                continue
            if typ == "BUF_X1" and conn.get("Z", "").startswith("out["):
                wires.append((conn["A"], conn["Z"], None))  # port buffer: modeled as a wire
                continue
            pins = lib["cells"][typ]["pins"]
            ins = {p: n for p, n in conn.items() if pins[p]["dir"] == "input"}
            outs = {p: (n, compile(py_function(pins[p]["func"]), typ, "eval"))
                    for p, n in conn.items() if pins[p]["dir"] == "output"}
            gates.append((typ, ins, outs))
        elif tok[0] == ".names":
            sig = tok[1:]
            rows = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("."):
                rows.append(lines[i].split()); i += 1
            if len(sig) == 1:
                wires.append((None, sig[0], bool(rows and rows[0] == ["1"])))
            elif len(sig) == 2 and rows == [["1", "1"]]:
                wires.append((sig[0], sig[1], None))
            else:
                raise ValueError(f"unsupported .names in {path}: {sig} {rows}")
    order = _levelize(gates, set(inputs) | {q for _, q in latches})
    return {"inputs": inputs, "outputs": outputs, "gates": [gates[j] for j in order],
            "latches": latches, "wires": wires}


def _levelize(gates, sources):
    drv = {}
    for j, (_, _, outs) in enumerate(gates):
        for n, _c in outs.values():
            drv[n] = j
    order, state = [], {}

    def visit(j):
        if state.get(j) == 2:
            return
        assert state.get(j) != 1, "combinational loop"
        state[j] = 1
        for n in gates[j][1].values():
            if n in drv:
                visit(drv[n])
        state[j] = 2
        order.append(j)
    for j in range(len(gates)):
        visit(j)
    return order


def _bit(n):
    return int(n[n.index("[") + 1:-1])


def run(net, lib, U, clr, T):
    """U, clr: operand/clear per cycle; T: reference register value at the start of each cycle (len N+1)."""
    N = len(U)
    V = lib["V"]
    val = {}
    for n in net["inputs"]:
        if n.startswith("in["):
            val[n] = ((U >> _bit(n)) & 1).astype(bool)
        elif n == "clr":
            val[n] = clr.astype(bool)
        else:
            val[n] = np.zeros(N, bool)
    for d, q in net["latches"]:
        val[q] = ((T[:N] >> _bit(q)) & 1).astype(bool)
    for src, dst, const in net["wires"]:
        if src is None:
            val[dst] = np.full(N, const)
    for typ, ins, outs in net["gates"]:
        env = {p: val[n] for p, n in ins.items()}
        for p, (n, code) in outs.items():
            val[n] = np.asarray(eval(code, {}, env), bool) & np.ones(N, bool)
    for src, dst, const in net["wires"]:
        if src is not None:
            val[dst] = val[src]
    # functional check: next-state net of every register equals the reference next state
    ok = all(np.array_equal(val[d], ((T[1:N + 1] >> _bit(q)) & 1).astype(bool)) for d, q in net["latches"])
    ok = ok and all(np.array_equal(val[o], ((T[:N] >> _bit(o)) & 1).astype(bool))
                    for o in net["outputs"] if o.startswith("out["))
    load = {}
    for typ, ins, _ in net["gates"]:
        for p, n in ins.items():
            load[n] = load.get(n, 0.0) + lib["cells"][typ]["pins"][p]["cap"]
    ff = lib["cells"]["DFF_X1"]
    for d, _q in net["latches"]:
        load[d] = load.get(d, 0.0) + ff["pins"]["D"]["cap"]
    # per-net toggle energy: driver internal energy + switched capacitance
    e_net, pin_e = {}, {}
    for typ, ins, outs in net["gates"]:
        pins = lib["cells"][typ]["pins"]
        for p, (n, _c) in outs.items():
            e_net[n] = pins[p]["e_int"] + 0.5 * load.get(n, 0.0) * V * V
        for p, n in ins.items():
            pin_e[n] = pin_e.get(n, 0.0) + pins[p]["e_int"]
    for _d, q in net["latches"]:
        e_net[q] = ff["pins"]["Q"]["e_int"] + 0.5 * load.get(q, 0.0) * V * V
    for n in net["inputs"]:
        e_net.setdefault(n, 0.5 * load.get(n, 0.0) * V * V)
    for d, _q in net["latches"]:
        pin_e[d] = pin_e.get(d, 0.0) + ff["pins"]["D"]["e_int"]
    e_clk = (ff["pins"]["CK"]["e_int"] * 2 + ff["pins"]["CK"]["cap"] * V * V) * len(net["latches"])
    # zero-delay energy
    tog0 = {n: int(np.count_nonzero(v[1:] != v[:-1])) for n, v in val.items()}
    e0 = sum(tog0[n] * (e_net.get(n, 0.0) + pin_e.get(n, 0.0)) for n in tog0)
    # unit-delay energy: transitions t -> t+1 for all t in parallel
    cur = {n: v[:-1].copy() for n, v in val.items()}
    for n in list(cur):  # sources switch at time 0
        if n in net["inputs"] or any(n == q for _d, q in net["latches"]):
            cur[n] = val[n][1:].copy()
    tog1 = {n: np.count_nonzero(cur[n] != val[n][:-1]) for n in cur if n in net["inputs"] or
            any(n == q for _d, q in net["latches"])}
    tog1 = {n: int(t) for n, t in tog1.items()}
    for _step in range(400):
        new = {}
        for typ, ins, outs in net["gates"]:
            env = {p: cur[n] for p, n in ins.items()}
            for p, (n, code) in outs.items():
                new[n] = np.asarray(eval(code, {}, env), bool) & np.ones(N - 1, bool)
        changed = False
        for n, v in new.items():
            t = int(np.count_nonzero(v != cur[n]))
            if t:
                tog1[n] = tog1.get(n, 0) + t
                changed = True
            cur[n] = v
        if not changed:
            break
    else:
        raise RuntimeError("unit-delay simulation did not settle")
    settled = all(np.array_equal(cur[n], val[n][1:]) for _t, _i, outs in net["gates"] for n, _c in outs.values())
    e1 = sum(t * (e_net.get(n, 0.0) + pin_e.get(n, 0.0)) for n, t in tog1.items())
    cyc = N - 1
    leak = sum(lib["cells"][g[0]]["leak_nW"] for g in net["gates"]) + len(net["latches"]) * ff["leak_nW"]
    return ok and settled, {"e_zero_fJ": e0 / cyc + e_clk, "e_unit_fJ": e1 / cyc + e_clk, "e_clock_fJ": e_clk,
                            "glitch_ratio": (e1 + 1e-12) / (e0 + 1e-12), "leak_nW": leak,
                            "n_ff": len(net["latches"]), "n_gates": len(net["gates"])}


def mac_trace(n=4000, K=144, signed=False, seed=11):
    """Operand and clear streams of a MAC: products of 8-bit values, cleared every K operations."""
    rng = np.random.default_rng(seed)
    if signed:
        a, b = rng.integers(-127, 128, n), rng.integers(-127, 128, n)
    else:
        a, b = rng.integers(0, 256, n), rng.integers(0, 256, n)
    U = (a * b) & 0xFFFF
    clr = (np.arange(n) % K) == 0
    return U.astype(np.int64), clr


def reference(step, U, clr):
    T = np.zeros(len(U) + 1, np.int64)
    for t in range(len(U)):
        T[t + 1] = 0 if clr[t] else int(step(np.array([T[t]]), np.array([U[t]]))[0]) & MASK24
    return T
