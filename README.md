# Carry Conservation in Approximate Accumulators

Research artifact for the manuscript *Carry Conservation in Approximate Accumulators: Quantizer Equivalence and an Entropy Bound* (Rei Li, Xinyi Li, and Ivy Li), prepared for submission to *Integration, the VLSI Journal*.

The paths below match the manuscript's Data availability statement.

## Contents

| Path | Contents |
|---|---|
| `python/accrate/` | Analysis, hardware-flow, and CNN/FIR scripts with their result files (`*.json`, `*.jsonl`) and run logs (`*.log`) |
| `python/accrate/*.pt` | Trained ResNet-20 weights: CIFAR-10 (seeds 0 and 1) and CIFAR-100 |
| `paper/integration_carry/` | LaTeX sources of the manuscript and supplementary material, with figures |
| `paper/integration_carry/qa/PREREGISTRATION_CIFAR100_20261030.md` | CIFAR-100 replication protocol. SHA-256 `a58846f76ececa377159e3c4eda8c9d8ee51dd955869200a37ed7b0569e400be` |
| Release asset `carry-conservation-netlists.zip` | Mapped Nangate45 netlists, ABC logs, and equivalence-check logs for the unsigned (`asic/`) and signed (`asic_s/`) studies. Unzip at the repository root. |

## Main scripts

| Topic | Scripts |
|---|---|
| Theory checks | `theory.py`, `pathwise_check.py`, `bound_property_test.py`, `entropy_bound_test.py`, `law_bound.py`, `threshold_quantizer.py`, `state_bound.py`, `machine_search.py` |
| Design census | `acc_designs.py`, `add_designs.py`, `evo_c2py.py`, `carry_census.py`, `census.py` |
| Synthesis (Yosys + ABC, Nangate45) | `asic_flow.py`, `run_hw_study.py`, `hw_analysis.py`, `prune_check.py`, `dominance.py`, `netsim.py`, `liberty.py` |
| Place and route (OpenROAD-flow-scripts) | `pnr_flow.py`, `pnr_analysis.py`, `make_pnr_tables.py` |
| FPGA (Vivado 2022.2, KV260) | `fpga_flow.py`, `fpga_analysis.py` |
| INT8 ResNet-20 | `resnet20.py`, `cnn_accumulate.py`, `analyze_cnn.py`, `cnn_energy.py`, `cnn_aux.py`, `tost.py` |
| CIFAR-100 replication | `cifar100_run.sh`, `cifar100_analysis.py`, `mse_sparsity_diag.py` |
| FIR on Kodak images | `fir_study.py`, `fir_hw.py` |
| Figures | `make_figures.py`, `make_submission_figures.py` |

## Requirements

- Python 3.12 with `numpy`, `scipy`, `matplotlib`, and `torch`.
- Hardware flows:
  - Yosys and ABC (OSS CAD Suite);
  - the `openroad/orfs` Docker image;
  - Icarus Verilog and OpenSTA;
  - Vivado 2022.2.
- Some flow scripts contain absolute tool and output paths for the authors' machine, such as `D:/tools/oss-cad-suite`, `D:/accrate_pnr`, and `D:/Xilinx/Vivado/2022.2`. Edit them before rerunning.

## External data (not redistributed)

- **EvoApproxLib v2022**: <https://github.com/ehw-fit/evoapproxlib>. The scripts expect it under `python/results/research_20261022/primary_sources/evoapproxlib_full_v2022`.
- **CIFAR-10 and CIFAR-100**: <https://www.cs.toronto.edu/~kriz/cifar.html>.
- **Kodak image set**: used by `fir_study.py`.

## Building the paper

```
cd paper/integration_carry
python build_bib.py
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
```

## License

The code is released under the MIT License (see `LICENSE`).

## Citation

Please cite the accompanying article; citation metadata are in `CITATION.cff`.
