"""Out-of-context Vivado implementation of every signed accumulator on the KV260 device (xck26).

Flow (identical for every design, Vivado 2022.2): synth_design -mode out_of_context
-flatten_hierarchy full; opt_design; place_design; route_design, at a 2.0 ns clock.
Reported: CLB LUTs, CLB registers, CARRY8 blocks, and the routed worst setup slack.
The RTL is the same acc.v used by pnr_flow.py (signed variant), minus the `default_nettype line.

  python fpga_flow.py run [-j 4]
  python fpga_flow.py collect      # -> fpga_study.json
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PNR = Path(os.environ.get("ACC_PNR_DIR", r"D:/accrate_pnr"))
OUT = Path(os.environ.get("ACC_FPGA_DIR", r"D:/accrate_fpga"))
VIVADO = r"D:/Xilinx/Vivado/2022.2/bin/vivado.bat"
PART = "xck26-sfvc784-2LV-c"
PERIOD = 2.0

TCL = r"""
set part {part}
foreach {{d top src}} [list {pairs}] {{
  set rpt {out}/$d.rpt
  if {{[file exists $rpt]}} continue
  if {{[catch {{
  create_project -in_memory -part $part
  read_verilog $src
  synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy full
  create_clock -period {period} -name clk [get_ports clk]
  opt_design
  place_design
  route_design
  set fh [open $rpt w]
  puts $fh "LUT [llength [get_cells -hier -filter {{PRIMITIVE_GROUP == LUT}}]]"
  puts $fh "FF [llength [get_cells -hier -filter {{PRIMITIVE_GROUP == REGISTER}}]]"
  puts $fh "CARRY8 [llength [get_cells -hier -filter {{REF_NAME == CARRY8}}]]"
  puts $fh "WNS [get_property SLACK [get_timing_paths -max_paths 1 -setup]]"
  close $fh
  report_utilization -file {out}/$d.util
  }} err]}} {{ puts "DESIGN_FAILED $d: $err" }}
  catch {{close_project}}
}}
"""


def designs():
    return sorted(p.name for p in (PNR / "s" / "relax").iterdir() if (p / "acc.v").exists())


def run(j=4):
    OUT.mkdir(parents=True, exist_ok=True)
    ds = designs()
    chunks = [ds[i::j] for i in range(j)]

    src = OUT / "src"
    src.mkdir(exist_ok=True)
    for d in ds:  # Vivado rejects ANSI port lists under `default_nettype none; the directive is lint only
        text = (PNR / "s" / "relax" / d / "acc.v").read_text()
        (src / f"{d}.v").write_text(text.replace("`default_nettype none", ""))

    def go(i):
        top = lambda d: json.loads((PNR / "s" / "relax" / d / "meta.json").read_text())["top"]  # noqa: E731
        pairs = " ".join(f"{d} {top(d)} {{{(src / f'{d}.v').as_posix()}}}" for d in chunks[i])
        tcl = OUT / f"batch{i}_{os.getpid()}.tcl"
        tcl.write_text(TCL.format(part=PART, pairs=pairs, out=OUT.as_posix(), period=PERIOD))
        return subprocess.run([VIVADO, "-mode", "batch", "-nojournal", "-log", str(OUT / f"batch{i}_{os.getpid()}.log"),
                               "-source", str(tcl)], cwd=OUT, capture_output=True, text=True).returncode
    with ThreadPoolExecutor(max_workers=j) as ex:
        print(list(ex.map(go, range(j))))


def collect():
    rows = []
    for d in designs():
        f = OUT / f"{d}.rpt"
        if not f.exists():
            continue
        r = {"design": d}
        r["wns"] = float(re.search(r"^WNS (\S+)", f.read_text(), re.M).group(1))
        util = (OUT / f"{d}.util").read_text()
        for key, pat in (("lut", "CLB LUTs"), ("ff", "CLB Registers"), ("carry8", "CARRY8")):
            r[key] = int(re.search(rf"^\|\s*{pat}\s*\|\s*(\d+)", util, re.M).group(1))
        r["fmax_mhz"] = 1e3 / (PERIOD - r["wns"])
        rows.append(r)
    (HERE / "fpga_study.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} of {len(designs())} designs implemented")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    else:
        collect()
