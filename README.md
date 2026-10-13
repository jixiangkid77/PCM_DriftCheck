# PCM_DriftCheck

Reproduction package for the study of **concurrent error detection on drifting
phase change memory crossbars**: a time-dependent tolerance band derived from
the fault-free statistics of column-group checksum residuals.

Under power-law conductance drift the classical checksum invariant of
algorithm-based fault tolerance survives on the crossbar, but its *decision
rule* does not: exact equality alarms on lawful drift, and a constant tolerance
loses its false-positive budget within one decade of retention. This repository
holds the experiments that quantify the failure, derive the replacement rule
`f(t) = theta0_hat * rho(t)`, and characterise where it stops working.

## Repository layout

```
PCM_DriftCheck/
├── src/pcm_drift/     shared library: drift_model, crossbar, encodings,
│                       checksum, faults, metrics, io_utils, model
├── experiments/       one entry script per experiment, E{n}_{name}.py + E{n}_config.py
├── logs/              E{n}_<runstamp>.log
├── figures/           make_paper_figs.py and the figures it produces
├── rtl/               E6 design files; E8.2 cost references (online_*.v)
├── tb/                E6 and E8.2 testbenches and vectors
└── synth/             yosys and OpenSTA scripts
```

Running an experiment writes its outputs to `results/E{n}/<runstamp>/`, one
directory per execution, never overwriting an earlier one. Those record
directories are produced locally and are not part of this repository; the log
of each run is, so the gate verdicts stay visible.

Two layers, deliberately separated. `src/pcm_drift` is the **instrument**: the
physics and the simulator, implemented once and imported by every experiment,
so that cross-experiment comparisons run on identical code paths. Everything
under `experiments/` is **protocol**: thin scripts that load a config, call the
library and write a run directory.

## Experiments

| id | script | question | formal runstamp |
|---|---|---|---|
| E1 | `E1_naive_checksum_failure.py` | how does a fixed threshold fail under drift, and how do the three encodings differ | `20260829_1239` |
| E2 | `E2_residual_model_validation.py` | does the closed-form residual variance law match simulation | `20260829_1344` |
| E3 | `E3_validity_domain_scan.py` | detection limits, the segmentation and scale criteria, budget backtest | `20260830_0124` |
| E4 | `E4_baseline_comparison.py` | the three decision rules against each other | `20260830_0150` |
| E5 | `E5_fault_class_coverage.py` | coverage by fault class and the sign-flip confusability edge | `20260830_0349` |
| E6 | `E6_checker_flow.py` | RTL of the digitized checker, bit-true behaviour, area and timing | `20260830_1309` |
| E8 | `E8_published_baselines.py` | the band against the published decision rules (exact equality, constant tolerance, mean tracking, online variance-adaptive thresholds) on the E3 ensembles and the E4 fault set | `20261011_0440` |
| E8.1 | `E8_1_transient.py` | exploratory: excess alarms after a fault appears, band vs online estimators | `20261011_0459` |
| E8.2 | `E8_2_online_hw.py` | RTL, bit-true check and synthesis of the online estimators as a cost reference | `20261011_0454` |

E8 reproduces the E4 rows bit for bit (gates G1, G2) and the E3 encoding-a band
rows (gate G4); its gate verdicts are in its log. E8.1 run `20261011_0444` used
a first-alarm metric that counts false alarms as detections and is superseded by
`20261011_0446`/`20261011_0459` (identical outputs). E8.2 run `20261011_0452`
preceded the addition of the E|r| variant and is superseded by `20261011_0454`.
E7 is a reserved number (online parameter estimation) and has not been run.

The runstamp column identifies the execution each reported number came from;
`logs/` carries the corresponding run log.

## Reproducing

```bash
conda activate pcm_sim          # python 3.10, numpy 2.0
pip install -e .                # or add src/ to PYTHONPATH

python experiments/E1_naive_checksum_failure.py
python experiments/E2_residual_model_validation.py
python experiments/E3_validity_domain_scan.py
python experiments/E4_baseline_comparison.py
python experiments/E5_fault_class_coverage.py
python experiments/E6_checker_flow.py --eda auto
python experiments/E8_published_baselines.py --e3 results/E3/20260830_0124 --e4 results/E4/20260830_0150
python experiments/E8_1_transient.py --e8 results/E8/20261011_0440
python experiments/E8_2_online_hw.py --e8 results/E8/20261011_0440
```

E6 and E8.2 additionally need Icarus Verilog, yosys and OpenSTA (E8.2:
Icarus Verilog and yosys, called directly). With yosys 0.33 the E6 netlist
`synth/out/drift_checker_8_4_c1.v` is reproduced byte for byte. The liberty file
`sky130_fd_sc_hd__tt_025C_1v80.lib` is **not** included here; take it from the
open sky130 PDK and point the synthesis scripts at it. On Windows these tools
were invoked through WSL2 (`--eda auto` resolves this).

The figure generator reads the record directories, so run the experiments
first; the paths below are the ones the six runs above produced:

```bash
python figures/make_paper_figs.py \
  --e1 results/E1/20260829_1239 --e2 results/E2/20260829_1344 \
  --e3 results/E3/20260830_0124 --e4 results/E4/20260830_0150 \
  --e5 results/E5/20260830_0349 --e6 results/E6/20260830_1309 \
  --e81 results/E8.1/20261011_0459 --out figures
```

A single figure, e.g. the construction schematic, which needs no record:

```bash
python figures/make_paper_figs.py --only fig1_construction
```

## Notes on reproducibility

Seeds are fixed, so every reported number is bit-reproducible across
independent runs on the same platform. Across platforms the only observed
difference is last-bit rounding of `exp` and `log` in libm: `eps_curve.csv`
agrees to about `1.5e-8` absolute, everything else is byte identical.

The model shares no fitted parameter with the simulator. Comparisons in E2 are
zero-fit over 432 grid cells.
