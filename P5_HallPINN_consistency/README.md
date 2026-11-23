# Data and code for "Beyond Data Fitting: Physical Consistency in Physics-Informed Hall Thruster Modeling"

Josh Adjedj and Subrata Roy, Applied Physics Research Group, University of Florida.

This record contains the input data, the trained physics-informed neural networks (PINNs), the training code and logs, and the network-independent analyses behind the results of the manuscript submitted to *Plasma Sources Science and Technology*. The operating point is the P5 Hall thruster at 300 V and 5.4 A.

## Contents

```
data/                    input data (experimental profiles and magnetic field)
reference_run/           how to reproduce the reference reconstruction (Sec. 5) with the code at the repository root
experiments/             constraint sequence and compatibility map (Sec. 7), one folder per training run
experiments/paper_checks network-independent diagnostics on the trained fields (Sec. 8): compatibility map, ion trajectories
experiments/bfield_2d_post  post-processing of the runs (energy budgets, momentum-implied currents, appendix figures)
analysis/                field-line test, uncertainty propagation and sensitivity (Sec. 8), digitization, figure scripts
MANIFEST.sha256          SHA-256 checksum of every file
```

### data/

| File | Content |
|---|---|
| `haas_gallimore_p5_5p4A_reconstruction.csv` | Training targets: analytic reconstruction of the internal profiles of $n_i$, $T_e$ and $V_p$ of Haas and Gallimore (AIAA-2000-3422, Figs. 5–9), 5 radii × 71 axial stations (x = 10–80 mm). This is the file every PINN run was trained on. |
| `haas_gallimore_p5_5p4A_digitized_points.csv` | Points extracted directly from the vector graphics of AIAA-2000-3422, Figs. 5–9 (raw marker positions, Sec. 8.5). |
| `haas_gallimore_p5_5p4A_digitized_grid.csv` | Same points on the 1 mm grid of the reconstruction (linear interpolation; at 12.5 mm two missing stations of $n_i$ and $T_e$ are filled linearly). |
| `p5_magnetic_field_measured.csv` | Measured vacuum magnetic field of the P5 (Haas, PhD thesis, University of Michigan, 2001, Figs. 5.41–5.60), radial and axial components at five radii, 1.6 and 3 kW. |
| `magnetic_field_fit/` | Divergence-free flux-function fit of the measured field (`field.py`, coefficients `field_main.json`) and the admissible field variants of Sec. 8.2 (`field_variants.*`). |

Columns of the profile files: `x_mm` (axial distance from the anode), `r_mm` (distance from the inner channel wall), `quantity` (`ni` in m⁻³, `Te` in eV, `Vp` in V), `value`, `unit`.

The original measurements belong to Haas and Gallimore and must be cited as such: J. M. Haas and A. D. Gallimore, AIAA Paper 2000-3422; J. M. Haas, PhD thesis, University of Michigan, 2001. The published PDF is not redistributed here; `analysis/digitization/extract_aiaa.py` reproduces the digitized points from it.

### reference_run/

The reference reconstruction of Sec. 5 (Gaussian magnetic field, isotropic closure, algebraic current closure, 15,000 Adam epochs, weight seed 7, collocation seed 2601) is produced by the code at the root of this repository (`code/`). Relative $L^2$ errors against the reconstructed profiles: $V_p$ 0.0173, $T_e$ 0.0730, $n_i$ 0.2048; discharge current 5.394 A. `reference_run/README.md` gives the command.

### experiments/

Each training run keeps its original folder name, because runs are warm-started from the final checkpoint of an earlier run (`../<previous run>/output/checkpoint.pt`). Each folder contains the PINN code (`code/`), the driver for that stage (`experiment.py`), the evaluation helpers (`analyze.py`, `diagnostics.py`), the magnetic field fit, the training log, the per-term loss history (`loss_terms.csv`), the diagnostics recorded during training (`progress_diagnostics.jsonl`), the final metrics when written, and the checkpoint used in the manuscript.

Runs reported in the constraint-compatibility map (Fig. 7 of the manuscript):

| Formulation in the manuscript | Folder | Checkpoint |
|---|---|---|
| Measured $B$ | `bfield_2d` | `output/checkpoint.pt` (epoch 6000) |
| Enlarged potential | `bfield_2d_phifix` | `output/checkpoint.pt` (6000) |
| No density data | `phifix_no_ni_data` | `output/checkpoint_epoch_6000.pt` |
| A: uncertainty weights + $p_e$ | `stageA_uw_pressure` | `output/checkpoint.pt` (6000) |
| B: + local field-aligned residual | `stageB3_par_continue` | `output/checkpoint.pt` (8000) |
| C: + current (5.4 A) | `stageC_current` | `output/checkpoint.pt` (6000) |
| Electron-only, local | `frozenC_electron` | `output/checkpoint.pt` (6000) |
| Electron-only, mobility-weighted | `frozenD_mobility` | `output/checkpoint.pt` (6000) |
| Electron-only, tensor + FLI | `frozenF_tensor_fli_local` | `output/checkpoint_epoch_8000.pt` |
| Coupled, tensor + FLI | `coupledG_tensor_fli` | `output/checkpoint_epoch_5500.pt` |

Intermediate runs included only because later runs are warm-started from them: `roy_current` (measured inflow and integral balances, Sec. 7.1), `ion_ramp`, `anode_bc` (anode potential 274.04 V), `stageB2_par_ramp`.

Warm-start chain: `roy_current` → `ion_ramp` → `anode_bc` → `bfield_2d` → `bfield_2d_phifix` → {`phifix_no_ni_data`, `stageA_uw_pressure` → `stageB2_par_ramp` → `stageB3_par_continue` → `stageC_current`}; `stageC_current` → `frozenC_electron` → `frozenD_mobility` → `frozenF_tensor_fli_local`; `coupledG_tensor_fli` restarts from `stageC_current` with the electron fields of `frozenD_mobility`.

Checkpoints are PyTorch dictionaries (`model_state_dict`, `domain`, `physics`, `weights`, optimizer and scheduler states, RNG states, `epochs_completed`, seeds). Load with `analyze.load_model(path)` from inside the run folder.

To re-run a stage: `cd experiments/<run> && python experiment.py`. Note that `verify_bases()` in `experiment.py` checks the SHA-256 of the base files listed in a `base_sha256.json` written on the original machine (absolute paths); this file is not included, so that check must be skipped when re-running. Training was done on CPU in float64; runs take 2–4 h each.

The four training logs of the electron-only and coupled runs end with `RuntimeError: postprocessing skipped (custom diagnostics in progress_diagnostics.jsonl)`. This is intentional: the generic post-processing was disabled for these runs and the diagnostics were recorded during training instead.

### experiments/paper_checks and experiments/bfield_2d_post

- `paper_checks/compat_map.py` rebuilds `compat_map.csv`/`.json` (the compatibility map) from `progress_diagnostics.jsonl` of each run and from the energy budgets and momentum-implied currents in `bfield_2d_post/`.
- `paper_checks/ion_trajectories.py <run folder> [checkpoint]` computes 30,000 ion trajectories with Miller charge exchange in the fields of a run (Sec. 8.6); outputs `ion_traj_<run>.json`.
- `paper_checks/independent_checks.py` is the field-line integral and parallel-current solve on the reconstructed profiles (Secs. 8.2–8.4).
- `bfield_2d_post/` contains the scripts and outputs for the ion energy budgets (`energy_*.json`), momentum-implied currents (`roy_<run>/current_compare.csv`), field-aligned terms (`eterms*.py`, `centrifugal.py`), and `make_appendix_figs.py`.

### analysis/

- `fieldline_sensitivity/reconstruction/` and `fieldline_sensitivity/digitized/`: the same scripts run on the two data sets. `independent_checks.py` (field-line mismatch, Monte Carlo with three error models, parallel current), `sensitivity.py` (radial interpolation schemes and axial probe-position uncertainty), `total_mc.py` (combined uncertainty, Table 6). Each folder contains its input profiles as `haas_gallimore_p5_5p4A.csv` and the JSON outputs.
- `digitization/extract_aiaa.py`: extraction of the data points from the vector figures of AIAA-2000-3422. Usage: `python extract_aiaa.py <AIAA-2000-3422.pdf> <raw_output.csv> <reconstruction.csv>`.
- `figures/`: scripts for the manuscript figures (`make_figs.py`: field-aligned terms, field-line mismatch, ion energy and trajectories; `compat_figure.py`: compatibility map; `figs_digitized.py`: Figs. 10–11).

## Reproducibility checks done before deposit

From this record, without any other file: `analysis/fieldline_sensitivity/*/independent_checks.py`, `sensitivity.py` and `total_mc.py` reproduce their JSON outputs exactly; `experiments/paper_checks/compat_map.py` reproduces `compat_map.csv` exactly; `ion_trajectories.py ../stageC_current` reproduces `ion_traj_stageC_current.json` to a relative difference below 1e-12.

## Software

Python 3, PyTorch (float64), NumPy, SciPy, Matplotlib, PyMuPDF (digitization only).

## Licence

Data and checkpoints: CC BY 4.0. Code: MIT (see `LICENSE`).

## Citation

Please cite the article (reference to be added on publication) and this record.
