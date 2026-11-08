"""Post-route Nangate45 evaluation of every accumulator with OpenROAD-flow-scripts.

Flow (identical for every design, run in the openroad/orfs Docker image):
  ORFS Yosys synthesis -> floorplan (fixed core utilization) -> global/detailed
  placement -> CTS -> global/detailed routing -> RC extraction (SPEF).
Each design is run at a relaxed and a tight clock target.  Area is the
post-route standard-cell area (fill and tap cells excluded, CTS buffers and
timing-repair buffers included).  Power is reported by OpenSTA from the
routed parasitics with switching activity annotated from a VCD.  The VCD comes
from a zero-delay gate-level simulation of the routed netlist on the same MAC
product trace used by asic_flow.py, and the simulated outputs are checked
cycle by cycle against the Python reference model.

  python pnr_flow.py prepare        # write per-design ORFS configs (u and s)
  python pnr_flow.py run [-j N]     # phase 1: ORFS in Docker
  python pnr_flow.py sim            # phase 2: gate-level simulation -> VCD (host iverilog)
  python pnr_flow.py power          # phase 3: OpenSTA post-route power in Docker
  python pnr_flow.py collect        # -> pnr_study.json
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PNR = Path(os.environ.get("ACC_PNR_DIR", r"D:/accrate_pnr"))  # large outputs; mounted at /work/pnr
IMAGE = "openroad/orfs:latest"
LIB = PNR / "NangateOpenCellLibrary_typical.lib"  # copied from the ORFS image
_OSS = Path(r"D:/tools/oss-cad-suite/bin")
IVERILOG, VVP = str(_OSS / "iverilog.exe"), str(_OSS / "vvp.exe")
_SIMENV = dict(os.environ, PATH=os.pathsep.join([str(_OSS), str(_OSS.parent / "lib"), os.environ.get("PATH", "")]))
# clock targets (ns): relaxed lets every design meet timing without repair,
# tight is about the fastest the exact accumulator closes, so timing repair is exercised.
# Ports are not buffered: in a MAC the accumulator's inputs and outputs are internal nets.
CLOCKS = {"relax": 1.50, "tight": 0.55}
# implementation variants, as the synthesis study's 18 mappings: every design gets the same set and
# the analysis takes the best routed result per design (job folder <clock>[-<cfg>])
CFGS = {"": {}, "a1": {"ABC_AREA": 1}, "u40": {"CORE_UTILIZATION": 40}, "u65": {"CORE_UTILIZATION": 65}}
N_TRACE, K_TRACE = 2000, 144


def designs():
    sys.path.insert(0, str(HERE))
    from run_hw_study import designs as ds
    return ds() + [f"tplus_{t}" for t in range(1, 9)]  # same 115 designs as hw_study_[us].json


def _wpath(p):
    """Path inside the container (PNR is mounted at /work/pnr)."""
    return "/work/pnr/" + Path(p).resolve().relative_to(PNR.resolve()).as_posix()


SDC = """current_design {top}
set clk_period {period}
create_clock -name core_clock -period $clk_period [get_ports clk]
create_clock -name vclk -period $clk_period
set_input_delay [expr $clk_period * 0.2] -clock vclk [all_inputs -no_clocks]
set_output_delay [expr $clk_period * 0.2] -clock vclk [all_outputs]
"""

CONFIG = """export DESIGN_NAME = {top}
export DESIGN_NICKNAME = {nick}
export PLATFORM = nangate45
export VERILOG_FILES = {src}
export SDC_FILE = {sdc}
export CORE_UTILIZATION = 50
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
export GPL_PLACE_IOS = 1
export SYNTH_REPEATABLE_BUILD = 1
export DONT_BUFFER_PORTS = 1
export PDN_TCL = $(PLATFORM_DIR)/../../designs/nangate45/gcd/grid_strategy-M1-M4-M7.tcl
"""


def pruned_rtl(acc_designs, d):
    """Corollary 1 at RTL level: the state-blind design with its low-state reads tied to zero and its
    low output bits dropped, so synthesis removes the low registers.  Upper outputs equal the twin's."""
    top, text, extra = acc_designs.rtl(d)
    k = acc_designs.evo_support(d[4:]) if d.startswith("evo_") else int(d.rsplit("_", 1)[1])
    W = acc_designs.W
    assert f"T[{k-1}:0]" in text and "assign out = T;" in text, d
    text = (text.replace(f"module {top}(", f"module {top}_prune(").replace(f"T[{k-1}:0]", f"{k}'d0")
            .replace("assign out = T;", f"assign out = {{T[{W-1}:{k}], {k}'d0}};"))
    return f"{top}_prune", text, extra


def blind_designs(signed):
    rows = json.loads((HERE / f"hw_study_{'s' if signed else 'u'}.json").read_text())
    return [r["design"] for r in rows if r["state_blind"] and r["k"] > 0 and not r["design"].startswith(
        ("exact", "round_", "trunc_", "tplus_"))]


def _write_job(job, top, src, tag, period, var, d, cfg):
    job.mkdir(parents=True, exist_ok=True)
    (job / "acc.v").write_text(src)
    (job / "constraint.sdc").write_text(SDC.format(top=top, period=period))
    conf = CONFIG.format(top=top, nick=f"{var}_{job.parent.name}_{d}", src=_wpath(job / "acc.v"),
                         sdc=_wpath(job / "constraint.sdc"))
    lines = [l for l in conf.splitlines() if not any(l.startswith(f"export {k} ") for k in CFGS[cfg])]
    lines += [f"export {k} = {v}" for k, v in CFGS[cfg].items()]  # override the defaults of CONFIG
    (job / "config.mk").write_text("\n".join(lines) + "\n")
    (job / "meta.json").write_text(json.dumps({"design": d, "top": top, "variant": var, "clock": tag,
                                               "cfg": cfg, "period_ns": period}))


def prepare_variant(signed):
    os.environ["ACC_SIGNED"] = "1" if signed else "0"
    sys.path.insert(0, str(HERE))
    import acc_designs
    assert acc_designs.SIGNED == signed
    var = "s" if signed else "u"
    jobs, srcs = [], []
    for d in designs():
        top, text, extra = acc_designs.rtl(d)
        srcs.append((d, top, text + "".join("\n" + Path(p).read_text() for p in extra)))
    for d in blind_designs(signed):
        top, text, extra = pruned_rtl(acc_designs, d)
        srcs.append((f"prune_{d}", top, text + "".join("\n" + Path(p).read_text() for p in extra)))
    for d, top, src in srcs:
        for tag, period in CLOCKS.items():
            for cfg in CFGS:
                job = PNR / var / (tag + (f"-{cfg}" if cfg else "")) / d
                if (job / "status").exists():
                    continue  # already routed; keep it
                _write_job(job, top, src, tag, period, var, d, cfg)
                jobs.append(job)
    if signed:
        jobs += prepare_mac(acc_designs)
    return jobs


MAC_DESIGNS = ("exact", "round_5", "round_6", "round_7", "round_8", "trunc_4", "trunc_6", "trunc_7",
               "evo_add8u_0E2", "evo_add8u_6QU", "evo_add8u_2LL", "loa_5")  # the CNN equivalence set
MAC_PERIOD = 2.0


def prepare_mac(acc_designs):
    """Signed 8x8 MAC: registered operands, multiplier, and the accumulator under test (area context)."""
    jobs = []
    for d in MAC_DESIGNS:
        top, text, extra = acc_designs.rtl(d)
        mac = (f"module mac_{d}(input clk, input clr, input signed [7:0] a, input signed [7:0] b, "
               f"output [23:0] out);\n  reg signed [7:0] ra, rb; reg rclr;\n"
               f"  always @(posedge clk) begin ra <= a; rb <= b; rclr <= clr; end\n"
               f"  wire signed [15:0] p = ra * rb;\n"
               f"  {top} u_acc(.clk(clk), .clr(rclr), .in(p), .out(out));\nendmodule\n")
        job = PNR / "m" / "relax" / d
        job.mkdir(parents=True, exist_ok=True)
        (job / "acc.v").write_text(mac + text + "".join("\n" + Path(p).read_text() for p in extra))
        (job / "constraint.sdc").write_text(SDC.format(top=f"mac_{d}", period=MAC_PERIOD))
        (job / "config.mk").write_text(CONFIG.format(top=f"mac_{d}", nick=f"m_relax_{d}",
                                                     src=_wpath(job / "acc.v"), sdc=_wpath(job / "constraint.sdc")))
        (job / "meta.json").write_text(json.dumps({"design": d, "top": f"mac_{d}", "variant": "m",
                                                   "clock": "relax", "period_ns": MAC_PERIOD}))
        jobs.append(job)
    return jobs


RUN_ONE = r"""#!/bin/bash
# $1 = job directory inside /work
job=$1; nick=$(basename $(dirname $(dirname $job)))_$(basename $(dirname $job))_$(basename $job)
[ -f $job/status ] && exit 0
source /OpenROAD-flow-scripts/env.sh > /dev/null 2>&1
W=/tmp/w/$nick
if make -C /OpenROAD-flow-scripts/flow DESIGN_CONFIG=$job/config.mk WORK_HOME=$W > $W.log 2>&1; then
  r=$W/results/nangate45/$nick/base
  cp $r/6_final.v $r/6_final.spef $r/6_final.odb $r/6_final.sdc $job/
  openroad -no_init -exit -no_splash /work/pnr/metrics.tcl > $job/metrics.log 2>&1 <<< "" || true
  echo ok > $job/status
else
  tail -60 $W.log > $job/flow_fail.log; echo fail > $job/status
fi
cp $W.log $job/flow.log 2>/dev/null
rm -rf $W $W.log
"""

METRICS = r"""set job $::env(JOB)
read_db $job/6_final.odb
read_liberty /OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_sdc $job/6_final.sdc
read_spef $job/6_final.spef
set block [ord::get_db_block]
set dbu [[ord::get_db_tech] getDbUnitsPerMicron]
set area 0.0; set ff 0; set buf 0; set n 0
foreach inst [$block getInsts] {
  set m [$inst getMaster]; set nm [$m getName]
  if {[string match FILLCELL* $nm] || [string match TAPCELL* $nm]} continue
  set area [expr {$area + double([$m getWidth]) * [$m getHeight] / $dbu / $dbu}]
  incr n
  if {[string match DFF* $nm]} {incr ff}
  if {[string match BUF* $nm] || [string match CLKBUF* $nm]} {incr buf}
}
set core [[$block getCoreArea] dx]
puts "METRIC cell_area_um2 $area"
puts "METRIC n_cells $n"
puts "METRIC n_ff $ff"
puts "METRIC n_buf $buf"
puts "METRIC wns_ns [sta::worst_slack -max]"
puts "METRIC whs_ns [sta::worst_slack -min]"
"""

POWER = r"""set job $::env(JOB)
read_db $job/6_final.odb
read_liberty /OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_sdc $job/6_final.sdc
read_spef $job/6_final.spef
read_vcd -scope tb/dut $job/act.vcd
report_power
"""


def _docker(cmd, mounts_extra=()):
    return subprocess.run(["docker", "run", "--rm", "-v", f"{PNR}:/work/pnr", IMAGE, "bash", "-c", cmd],
                          capture_output=True, text=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))


def _jobs():
    return sorted(p.parent for p in PNR.glob("[usm]/*/*/config.mk"))


def run(jobs=None, j=12):
    (PNR / "run_one.sh").write_bytes(RUN_ONE.replace("\r\n", "\n").encode())
    (PNR / "metrics.tcl").write_bytes(METRICS.encode())
    jobs = jobs or _jobs()
    lst = PNR / "joblist.txt"
    lst.write_bytes(("\n".join(_wpath(p) for p in jobs) + "\n").encode())
    # metrics.tcl reads JOB from the environment
    cmd = (f"sed -i 's#openroad -no_init#JOB=$job openroad -no_init#' /work/pnr/run_one.sh; "
           f"mkdir -p /tmp/w; xargs -P {j} -n 1 bash /work/pnr/run_one.sh < /work/pnr/joblist.txt")
    r = _docker(cmd)
    print(r.stdout[-2000:], r.stderr[-2000:])


# ---------------------------------------------------------------- phase 2: gate-level VCD

def cell_models(out):
    sys.path.insert(0, str(HERE))
    from liberty import _parse, _tokens
    toks = _tokens(LIB.read_text())
    lib, _ = _parse(toks, toks.index("{") + 1)

    def vexpr(f):
        return f.replace("!", "~").replace("*", "&").replace("+", "|")
    mods = []
    for g in lib["groups"]:
        if g.get("type") != "cell":
            continue
        name = g["args"][0]
        pins = [p for p in g["groups"] if p["type"] == "pin"]
        seq = [p for p in g["groups"] if p["type"] in ("ff", "latch")]
        ports, body = [], []
        for p in pins:
            d = p["attrs"].get("direction", "input")
            ports.append(f"{'output' if d == 'output' else 'input'} {p['args'][0]}")
        for s in seq:
            iq, iqn = s["args"][0], s["args"][1]
            body.append(f"  reg {iq}; wire {iqn} = ~{iq};")
            a = s["attrs"]
            if s["type"] == "ff":
                clk = a["clocked_on"].replace("!", "~")
                ev = [f"posedge {clk}" if not clk.startswith("~") else f"negedge {clk[1:]}"]
                clr, pre = a.get("clear"), a.get("preset")
                for sig in (clr, pre):
                    if sig:
                        ev.append(f"negedge {sig[1:]}" if sig.startswith("!") else f"posedge {sig}")
                stmt = f"{iq} <= {vexpr(a['next_state'])};"
                if pre:
                    stmt = f"if ({vexpr(pre)}) {iq} <= 1'b1; else {stmt}"
                if clr:
                    stmt = f"if ({vexpr(clr)}) {iq} <= 1'b0; else {stmt}"
                body.append(f"  always @({' or '.join(ev)}) {stmt}")
            else:
                body.append(f"  always @* if ({vexpr(a['enable'])}) {iq} = {vexpr(a['data_in'])};")
        for p in pins:
            f = p["attrs"].get("function")
            if p["attrs"].get("direction") == "output" and f:
                body.append(f"  assign {p['args'][0]} = {vexpr(f)};")
        mods.append(f"module {name}({', '.join(ports)});\n" + "\n".join(body) + "\nendmodule\n")
    Path(out).write_text("`timescale 1ps/1ps\n" + "\n".join(mods))


TB = """`timescale 1ps/1ps
module tb;
  reg clk = 0, clr = 1; reg [15:0] in = 0; wire [23:0] out;
  reg [16:0] mem [0:{n}];
  integer i, fo;
  {top} dut(.clk(clk), .clr(clr), .in(in), .out(out));
  always #{half} clk = ~clk;
  initial begin
    $readmemh("{trace}", mem);
    fo = $fopen("{outf}", "w");
    {{clr, in}} = mem[0];
    @(posedge clk); #1;
    $dumpfile("{vcd}"); $dumpvars(0, dut);
    for (i = 1; i <= {n}; i = i + 1) begin
      #{drive}; {{clr, in}} = mem[i];
      @(posedge clk); #1; $fwrite(fo, "%h\\n", out);
    end
    $fclose(fo); $finish;
  end
endmodule
"""


def _trace(signed):
    sys.path.insert(0, str(HERE))
    import netsim
    U, clr = netsim.mac_trace(n=N_TRACE, K=K_TRACE, signed=signed)
    return U, clr


def sim_job(job, models):
    meta = json.loads((job / "meta.json").read_text())
    if not (job / "6_final.v").exists() or (job / "sim_ok").exists():
        return job, None
    signed = meta["variant"] == "s"
    U, clr = _trace(signed)
    words = [(1 << 16) | 0] + [(int(c) << 16) | int(u) for u, c in zip(U, clr)]
    (job / "trace.hex").write_text("\n".join(f"{w:05x}" for w in words) + "\n")
    period = int(round(meta["period_ns"] * 1000))
    tb = TB.format(n=N_TRACE, top=meta["top"], half=period // 2, trace="trace.hex", outf="sim_out.txt",
                   vcd="act.vcd", drive=period // 2 - 1)
    (job / "tb.v").write_text(tb)
    r = subprocess.run([IVERILOG, "-g2012", "-o", "sim.vvp", "tb.v", "6_final.v", str(models)],
                       cwd=job, capture_output=True, text=True, env=_SIMENV)
    if r.returncode:
        return job, "compile: " + r.stderr[-800:]
    r = subprocess.run([VVP, "-n", "sim.vvp"], cwd=job, capture_output=True, text=True, env=_SIMENV)
    if r.returncode:
        return job, "run: " + r.stderr[-800:]
    (job / "sim.vvp").unlink()
    # reference: same step model as asic_flow.py (workers inherit ACC_SIGNED from sim())
    import acc_designs
    import netsim
    assert acc_designs.SIGNED == signed
    d = meta["design"]
    if d.startswith("prune_"):
        from theory import low_tables
        k, _z, c = low_tables(d[6:])
        g = c[0].astype(np.int64)  # state-blind: c(x, u) = g(u)
        q = 1 << k
        step = lambda T, V: ((((T >> k) + (V >> k) + g[V & (q - 1)]) << k)) & acc_designs.MASK  # noqa: E731
    else:
        step = acc_designs.step_fn(d)
    T = netsim.reference(step, acc_designs.operand24(U), clr)
    got = [int(x, 16) if "x" not in x else -1 for x in (job / "sim_out.txt").read_text().split()]
    ok = got == [int(t) for t in T[1:]]
    if ok:
        (job / "sim_ok").write_text("ok")
    return job, None if ok else f"mismatch at {next(i for i, (a, b) in enumerate(zip(got, T[1:])) if a != b)}"


def sim(j=16):
    models = PNR / "nangate45_models.v"
    cell_models(models)
    # one process per variant so acc_designs.SIGNED is fixed at import; threads run the simulators
    for var in ("u", "s"):
        subprocess.run([sys.executable, __file__, "_sim", var, str(j)], check=True,
                       env=dict(os.environ, ACC_SIGNED="1" if var == "s" else "0"))


def _sim_variant(var, j):
    models = PNR / "nangate45_models.v"
    jobs = [p for p in _jobs() if p.parts[-3] == var and (p / "status").exists()
            and (p / "status").read_text().strip() == "ok"]
    bad = []
    with ThreadPoolExecutor(max_workers=j) as ex:
        for job, err in ex.map(lambda p: sim_job(p, models), jobs):
            if err:
                bad.append((str(job), err))
    print(f"{var}: {len(jobs)} routed jobs, simulation failures: {len(bad)}")
    for b in bad[:20]:
        print(b)


def power(j=12):
    (PNR / "power.tcl").write_bytes(POWER.encode())
    jobs = [p for p in _jobs() if (p / "sim_ok").exists() and not (p / "power.log").exists()]
    lst = PNR / "powerlist.txt"
    lst.write_bytes(("\n".join(_wpath(p) for p in jobs) + "\n").encode())
    cmd = ("source /OpenROAD-flow-scripts/env.sh >/dev/null 2>&1; "
           f"xargs -P {j} -I{{}} sh -c 'JOB={{}} openroad -no_init -no_splash -exit /work/pnr/power.tcl "
           "> {}/power.log 2>&1' < /work/pnr/powerlist.txt")
    r = _docker(cmd)
    print(r.stdout[-1500:], r.stderr[-1500:])


def _power_total(text):
    m = re.search(r"^Total\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)", text, re.M)
    return None if m is None else dict(zip(("internal_W", "switching_W", "leakage_W", "total_W"),
                                           map(float, m.groups())))


def collect():
    rows = []
    for job in _jobs():
        meta = json.loads((job / "meta.json").read_text())
        st = (job / "status").read_text().strip() if (job / "status").exists() else "missing"
        r = dict(meta, status=st, sim_ok=(job / "sim_ok").exists())
        if (job / "metrics.log").exists():
            for k, v in re.findall(r"^METRIC (\S+) (\S+)", (job / "metrics.log").read_text(), re.M):
                r[k] = float(v)
        if (job / "power.log").exists():
            p = _power_total((job / "power.log").read_text())
            if p:
                r.update(p)
                r["energy_fJ_per_op"] = p["total_W"] * meta["period_ns"] * 1e-9 * 1e15
        rows.append(r)
    (HERE / "pnr_study.json").write_text(json.dumps(rows, indent=1))
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"{len(rows)} runs, {ok} routed, {sum(r['sim_ok'] for r in rows)} simulated, "
          f"{sum('total_W' in r for r in rows)} with power")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        n = 0
        for s in (False, True):
            n += len(subprocess.run([sys.executable, __file__, "_prep", str(int(s))], check=True,
                                    capture_output=True, text=True).stdout.split())
        print(f"{n} jobs")
    elif cmd == "_prep":
        print(" ".join(p.name for p in prepare_variant(sys.argv[2] == "1")))
    elif cmd == "run":
        sel = [Path(a) for a in sys.argv[2:] if not a.startswith("-")]
        run(sel or None)
    elif cmd == "sim":
        sim()
    elif cmd == "_sim":
        _sim_variant(sys.argv[2], int(sys.argv[3]))
    elif cmd == "power":
        power()
    elif cmd == "collect":
        collect()
