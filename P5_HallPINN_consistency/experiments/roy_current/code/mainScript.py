"""
Final P5 driver used for the paper archive.

This file is self-contained: the v26-derived domain/physics builder, model
builder, constants, history setup, and deterministic summary have been merged
directly into this script so it no longer depends on mainScript_v26_kiscale1.py.
"""

import json
import math
import os
import sys
import time

from integrated_constraints import mean_current_A

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache")

# Make sibling imports work whether launched as a script or as a module.
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np
import torch

from model import PINN
from utils import (
    compute_all_losses,
    compute_auto_weights,
    compute_collision_frequencies,
    compute_ion_neutral_frequency,
    compute_data_loss,
    compute_Si,
    compute_validation_residuals,
    create_output_folders,
    data_loss_l2_relative,
    evaluate_on_grid,
    load_haas_gallimore_data,
    sample_boundary_points,
    sample_domain_points,
    save_centerline_profiles,
    save_combined_plot,
    save_data_comparison_plots,
    save_global_metrics,
    save_individual_plots,
    save_loss_plot,
    save_radial_profiles,
    save_residual_maps,
)


torch.set_default_dtype(torch.float64)


LR = 1.0e-4
N_DOMAIN = 10000
N_INLET = 1000
N_WALL = 1000
SEED = 2601
NEUTRAL_INLET_FROM_MASS_FLOW = True
ALPHA_B = 1.0 / 200.0
V24_FINAL_LOSS_REF = 1.2258
LOSS_EXPLOSION_LIMIT = 10.0 * V24_FINAL_LOSS_REF


def build_domain_physics():
    e_phys = 1.602e-19
    eps0 = 8.854e-12
    mi = 2.180e-25
    me = 9.109e-31
    sigma0 = 3.6e-20
    Ei_eV = 12.1
    phi_ref = Ei_eV
    V_ref = math.sqrt(Ei_eV * e_phys / mi)
    Gamma_ref = 1.0e23
    n_ref = Gamma_ref / V_ref
    sigma_ref = sigma0 * math.sqrt(mi / me)
    nu_ref = sigma_ref * Gamma_ref
    L_canal = 0.038
    r_in_phys, r_out_phys = 0.060, 0.085
    S_ref = n_ref * V_ref / L_canal
    k_ref = V_ref / (L_canal * n_ref)
    alpha_p = (e_phys * n_ref * L_canal**2) / (eps0 * phi_ref)
    A_channel = math.pi * (r_out_phys**2 - r_in_phys**2)
    JT_norm = (5.4 / A_channel) / (e_phys * n_ref * V_ref)

    domain = {
        "x_min": 0.0, "x_max": 1.0,
        "r_min": r_in_phys / L_canal,
        "r_max": r_out_phys / L_canal,
        "phi_left": 300.0 / phi_ref,
        "phi_right": 116.2 / phi_ref,
    }
    physics = {
        "e_phys": e_phys,
        "me": me,
        "mi": mi,
        "mn": mi,
        "alpha": alpha_p,
        "n_ref": n_ref,
        "V_ref": V_ref,
        "phi_ref": phi_ref,
        "nu_ref": nu_ref,
        "tau_ref": L_canal / V_ref,
        "S_ref": S_ref,
        "k_ref": k_ref,
        "L_canal": L_canal,
        "eps_i": Ei_eV / phi_ref,
        "Vn": 300.0 / V_ref,
        "delta": 0.1,
        "Vix_inlet": 0.50,
        "nn_exit_target": 0.05,
        "ni_inlet": 0.003,
        "nn_inlet": 1.0,
        "Vetheta_inlet": 0.0,
        "JT_norm": JT_norm,
        "e_over_me": e_phys / me,
        "coulomb_log": 10.0,
        "sigma_en": 2.5e-19,
        "alpha_B": ALPHA_B,
        "Tn_eV": 0.025,
        "Ti_eV": 0.025,
        "energy_alpha": 1.0,
        "energy_scale": 1.0,
        "Te_base": 3.0 / phi_ref,
        "Te_peak": 27.0 / phi_ref,
        "x0_Te": 0.55,
        "sigma_x_Te": 0.15,
        "r0_Te": 0.5 * (r_in_phys + r_out_phys) / L_canal,
        "sigma_r_Te": 0.11842105263157895,
        "B_max": 0.020,
        "x_B": 0.75,
        "sigma_B": 0.30,
        "Di": 0.0,
        "ki_floor_eV": 3.5,
        "ki_softplus_beta": 8.0,
        "ki_scale": 1.0,
    }
    return domain, physics


def build_model(domain, physics, device):
    return PINN(
        input_dim=2,
        output_dim=10,
        hidden_dim=128,
        num_hidden_layers=4,
        phi_left=domain["phi_left"],
        phi_right=domain["phi_right"],
        x_min=domain["x_min"], x_max=domain["x_max"],
        r_min=domain["r_min"], r_max=domain["r_max"],
        delta=physics["delta"],
        Vix_inlet=physics["Vix_inlet"],
        ni_inlet=physics["ni_inlet"],
        nn_inlet=physics["nn_inlet"],
        A_ex=1000.0,
        Te_base=physics["Te_base"],
        Te_peak=physics["Te_peak"],
        x0_Te=physics["x0_Te"],
        sigma_x_Te=physics["sigma_x_Te"],
        r0_Te=physics["r0_Te"],
        sigma_r_Te=physics["sigma_r_Te"],
        eta_max=1.0e-4,
    ).to(device)


def init_history():
    keys = [
        "total", "poisson", "ion_cont", "neut_cont", "elec_cont", "ion_mom_x",
        "ni_inlet", "nn_inlet", "Vetheta_inlet", "Vir_wall",
        "Ver_wall", "integrated_current", "Id_mean_A", "Si_mean", "nn_exit",
        "smooth_ni", "smooth_nn", "eta_abs_mean", "eta_abs_max",
        "ni_mean", "ni_std",
    ]
    return {k: [] for k in keys}


RUN_NAME            = "run_d_alpha1over200_vetheta2000_seed7"
OUTPUT_DIR          = os.path.join(_parent, "output")
CKPT_IN             = ""
CKPT_OUT            = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_checkpoint.pt")
SUMMARY_OUT         = os.path.join(OUTPUT_DIR, f"VALIDATION_{RUN_NAME}.md")

EPOCHS_START        = 0
EPOCHS_END          = 15000
EPOCHS_TO_RUN       = EPOCHS_END - EPOCHS_START
LBFGS_STEPS         = 0
# Env-var override (no config.json mutation) — used by the sanity-test harness.
_epochs_override = os.environ.get("EPOCHS_OVERRIDE")
if _epochs_override is not None:
    EPOCHS_TO_RUN = int(_epochs_override)
    print(f"EPOCHS_TO_RUN overridden via env: {EPOCHS_TO_RUN}", flush=True)
_lbfgs_override = os.environ.get("LBFGS_STEPS_OVERRIDE")
if _lbfgs_override is not None:
    LBFGS_STEPS = int(_lbfgs_override)
    print(f"LBFGS_STEPS overridden via env: {LBFGS_STEPS}", flush=True)
# Env-var overrides for sanity / smoke runs that should not overwrite the
# production v26_kiscale1 checkpoint / summary.
CKPT_IN     = os.environ.get("CKPT_IN_OVERRIDE", CKPT_IN)
OUTPUT_DIR  = os.environ.get("OUTPUT_OVERRIDE", OUTPUT_DIR)
CKPT_OUT    = os.environ.get("CKPT_OUT_OVERRIDE", CKPT_OUT)
SUMMARY_OUT = os.environ.get("SUMMARY_OVERRIDE", SUMMARY_OUT)
INIT_SEED   = int(os.environ.get("SEED_OVERRIDE", SEED))
OPTIMIZER_OVERRIDE = os.environ.get("OPTIMIZER_OVERRIDE", "ADAM").strip().upper()
SMOOTH_NI_OVERRIDE = os.environ.get("SMOOTH_NI_OVERRIDE")
_centerline_data_env = os.environ.get("CENTERLINE_DATA_R_MM")
CENTERLINE_DATA_R_MM = (
    float(_centerline_data_env) if _centerline_data_env not in (None, "") else None
)
MONITOR_EVERY       = 500
PRINT_EVERY         = 250

# Hard-stop thresholds (user-supplied).
STOP_ETA_SAT_95     = 0.85
# V3 free-flux starts from lower normalized ni than v26; this catches true
# collapse while allowing the requested non-monotone hump to develop.
STOP_NI_PEAK_MIN    = 0.005

# Grid for monitoring snapshots (matches the audit grid; cheap on CPU).
MON_NX = 200
MON_NR = 80


def _grad(u, var):
    return torch.autograd.grad(u.sum(), var, create_graph=True, retain_graph=True)[0]


def report_ni_centerline_init(model, domain, device):
    r_center = 0.5 * (domain["r_min"] + domain["r_max"])
    x_probe = torch.linspace(
        domain["x_min"], domain["x_max"], 11,
        device=device, dtype=torch.float64,
    )
    r_probe = torch.full_like(x_probe, r_center)
    with torch.no_grad():
        _, ni, _, _, _, _, _, _, _, _, _ = model(x_probe, r_probe)
    pairs = [
        f"x={float(x.item()):.2f}:ni={float(n.item()):.5f}"
        for x, n in zip(x_probe, ni)
    ]
    print("Init ni centerline shape (before training/checkpoint load):", flush=True)
    print("  " + " | ".join(pairs), flush=True)


def report_ion_friction_init(model, domain, physics, device):
    x0 = torch.tensor([0.55], device=device, dtype=torch.float64, requires_grad=True)
    r0 = torch.tensor(
        [0.5 * (domain["r_min"] + domain["r_max"])],
        device=device, dtype=torch.float64, requires_grad=True,
    )
    phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, *_ = model(x0, r0)
    phi_x = _grad(phi, x0)
    Si = compute_Si(nn_, ne, Te, physics)
    coll = compute_collision_frequencies(ne, nn_, Te, physics)
    tau_ref = physics.get("tau_ref", physics["L_canal"] / physics["V_ref"])
    nu_in_hat = compute_ion_neutral_frequency(nn_, Vix, physics) * tau_ref
    ion_factor = (physics["mn"] / physics["mi"]) * nu_in_hat
    electric_x = ni * phi_x
    electron_drag_force_x = ni * (physics["me"] / physics["mi"]) * coll["nu_total"] * tau_ref * (Vex - Vix)
    electron_drag_residual_x = -electron_drag_force_x
    neutral_friction_x = ni * ion_factor * (Vix - physics["Vn"])
    birth_momentum_x = -Si * physics["Vn"]
    total_friction_x = electron_drag_residual_x + neutral_friction_x + birth_momentum_x
    ratio = total_friction_x.detach().abs() / torch.clamp(electric_x.detach().abs(), min=1e-30)
    print(
        "Init conservative ion-friction sign check at x=0.55 centerline: "
        f"Vix={float(Vix.item()):.6g}, Vn={physics['Vn']:.6g}, "
        f"nu_in_hat={nu_in_hat.item():.6g}, "
        f"neutral_friction_residual_x={float(neutral_friction_x.item()):.6e}, "
        f"birth_momentum_residual_x={float(birth_momentum_x.item()):.6e}, "
        f"electron_drag_residual_x={float(electron_drag_residual_x.item()):.6e}, "
        f"total_drag_residual_x={float(total_friction_x.item()):.6e}, "
        f"ni*dphi/dx={float(electric_x.item()):.6e}, "
        f"|friction/electric|={float(ratio.item()):.6g}. "
        "Positive neutral-friction residual at Vix>Vn corresponds to a negative physical drag force.",
        flush=True,
    )


def report_phi_bc_init(model, domain, device):
    r_probe = torch.linspace(
        domain["r_min"], domain["r_max"], 9,
        device=device, dtype=torch.float64,
    )
    x_left = torch.full_like(r_probe, domain["x_min"])
    x_right = torch.full_like(r_probe, domain["x_max"])
    with torch.no_grad():
        phi_left, *_ = model(x_left, r_probe)
        phi_right, *_ = model(x_right, r_probe)
    left_err = torch.max(torch.abs(phi_left - domain["phi_left"])).item()
    right_err = torch.max(torch.abs(phi_right - domain["phi_right"])).item()
    print(
        "Init phi hard-BC check: "
        f"left={float(phi_left[0]):.12g} target={domain['phi_left']:.12g} "
        f"max_err={left_err:.3e}; "
        f"right={float(phi_right[0]):.12g} target={domain['phi_right']:.12g} "
        f"max_err={right_err:.3e}",
        flush=True,
    )


def report_phi_shape_init(model, domain, physics, device):
    r_center = 0.5 * (domain["r_min"] + domain["r_max"])
    x_probe = torch.linspace(
        domain["x_min"], domain["x_max"], 11,
        device=device, dtype=torch.float64,
    )
    r_probe = torch.full_like(x_probe, r_center)
    xi = (x_probe - domain["x_min"]) / (domain["x_max"] - domain["x_min"])
    phi_linear = domain["phi_left"] + (domain["phi_right"] - domain["phi_left"]) * xi
    cap = xi * (1.0 - xi) * float(model.phi_shape_amp)
    with torch.no_grad():
        phi, _, Vix, _, _, _, _, _, _, _, _ = model(x_probe, r_probe)
    pairs = [
        f"x={float(x.item()):.2f}:phi={float(p.item() * physics['phi_ref']):.1f}V"
        for x, p in zip(x_probe, phi)
    ]
    print("Init phi centerline (A=25 correction active, raw0 at init):", flush=True)
    print("  " + " | ".join(pairs), flush=True)
    idx_055 = int(torch.argmin(torch.abs(x_probe - 0.55)).item())
    print(
        "Phi shape capacity check: "
        f"at x~0.55 linear={float(phi_linear[idx_055] * physics['phi_ref']):.2f}V, "
        f"max correction={float(cap[idx_055] * physics['phi_ref']):.2f}V "
        "(HG needs about +65.6V at x=0.55).",
        flush=True,
    )
    print(
        "Init Vix Bohm-referenced check: "
        f"Vix(x=0)={float(Vix[0]):.6g}, Vix(x~0.55)={float(Vix[idx_055]):.6g}, "
        f"Vix(exit)={float(Vix[-1]):.6g}, "
        f"anode target={getattr(model, 'Vix_anode_min', physics['Vix_inlet']):.6g}, "
        f"old Bohm={physics['Vix_inlet']:.6g}.",
        flush=True,
    )


def _evaluate_monitor(model, domain, physics, device):
    """Run the 4 monitored quantities on a deterministic 200x80 grid."""
    # ni on centerline: cheap 1-D evaluation, no autograd
    r_center = 0.5 * (domain["r_min"] + domain["r_max"])
    x_cl = torch.linspace(
        domain["x_min"], domain["x_max"], MON_NX,
        device=device, dtype=torch.float64,
    )
    r_cl = torch.full_like(x_cl, r_center)
    with torch.no_grad():
        _, ni_cl, _, _, _, _, _, _, _, _, _ = model(x_cl, r_cl)
    ni_cl_np = ni_cl.cpu().numpy()
    ni_peak_idx = int(np.argmax(ni_cl_np))
    ni_peak_value = float(ni_cl_np[ni_peak_idx])
    ni_peak_x = float(x_cl[ni_peak_idx].item())

    # eta saturation on 2-D grid: forward pass only, no grad
    x_lin = torch.linspace(
        domain["x_min"], domain["x_max"], MON_NX,
        device=device, dtype=torch.float64,
    )
    r_lin = torch.linspace(
        domain["r_min"], domain["r_max"], MON_NR,
        device=device, dtype=torch.float64,
    )
    XX, RR = torch.meshgrid(x_lin, r_lin, indexing="ij")
    with torch.no_grad():
        _, _, _, _, _, _, _, _, _, _, eta = model(XX.reshape(-1), RR.reshape(-1))
    eta_np = eta.cpu().numpy().reshape(MON_NX, MON_NR)
    eta_max_v = float(model.eta_max)
    eta_abs = np.abs(eta_np)
    sat_frac_95 = float(np.mean(eta_abs > 0.95 * eta_max_v))
    sat_frac_50 = float(np.mean(eta_abs > 0.5  * eta_max_v))

    # Residual RMS on the same 2-D grid (uses autograd).
    residuals = compute_validation_residuals(
        model, domain, physics, nx=MON_NX, nr=MON_NR, device=device
    )

    return dict(
        ni_peak_value=ni_peak_value,
        ni_peak_x=ni_peak_x,
        sat_frac_95=sat_frac_95,
        sat_frac_50=sat_frac_50,
        eta_min=float(eta_np.min()),
        eta_max_val=float(eta_np.max()),
        eta_mean=float(eta_np.mean()),
        IonCont_RMS=residuals["IonCont_RMS"],
        NeutCont_RMS=residuals["NeutCont_RMS"],
        Poisson_RMS=residuals["Poisson_RMS"],
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INIT_SEED)
    print(f"Using device: {device}", flush=True)
    print(
        f"Init seed before build_model = {INIT_SEED}; "
        f"collocation/boundary seed = {SEED}.",
        flush=True,
    )
    print(
        f"Training {RUN_NAME}: from_scratch={not bool(CKPT_IN)}, checkpoint={CKPT_IN or '<none>'}; "
        f"epoch {EPOCHS_START} -> {EPOCHS_START + EPOCHS_TO_RUN}. "
        f"{OPTIMIZER_OVERRIDE} epochs followed by {LBFGS_STEPS} L-BFGS steps. "
        f"v26 collocation counts are reused without resampling.",
        flush=True,
    )

    # 1) Rebuild domain/physics and apply continuity-experiment overrides.
    domain, physics = build_domain_physics()
    physics["ni_inlet"] = 4.64e17 / physics["n_ref"]
    physics["Vn"] = 450.0 / physics["V_ref"]
    physics["nn_exit_target"] = -1.0
    physics["ki_taper_center_eV"] = 5.0
    physics["ki_taper_width_eV"] = 0.35
    physics["ion_neutral_rate_model"] = "Miller2002_local_relative_speed_thermal_floor"
    physics["energy_alpha"] = 2.5
    e_phys = physics["e_phys"]
    L_canal = physics["L_canal"]
    r_in_phys = domain["r_min"] * L_canal
    r_out_phys = domain["r_max"] * L_canal
    A_channel = math.pi * (r_out_phys**2 - r_in_phys**2)
    physics["A_channel"] = A_channel
    physics["NEUTRAL_INLET_FROM_MASS_FLOW"] = NEUTRAL_INLET_FROM_MASS_FLOW
    physics["nn_inlet"] = 0.164145 if NEUTRAL_INLET_FROM_MASS_FLOW else 1.0
    physics["A_ex"] = 1000.0
    physics["integrated_axial_stations"] = 40
    physics["integrated_radial_GL"] = 32
    physics["integrated_axial_GL_per_interval"] = 8
    print(f"Measured-flow inlet flag={NEUTRAL_INLET_FROM_MASS_FLOW}; nn_inlet={physics['nn_inlet']:.9g}", flush=True)
    physics["JT_target_A"] = 5.4
    physics["JT_A_scale"] = e_phys * physics["n_ref"] * physics["V_ref"] * A_channel
    physics["JT_norm"] = physics["JT_target_A"] / physics["JT_A_scale"]
    print(
        f"Continuity experiment overrides: ni_inlet={physics['ni_inlet']:.6g} "
        f"({physics['ni_inlet'] * physics['n_ref']:.3e} m^-3), "
        f"Vn={physics['Vn'] * physics['V_ref']:.1f} m/s, "
        "nu_in=local nn*sigma_cex(E)*max(abs(Vix-Vn),v_thermal), "
        "nu_in_hat=nu_in*tau_ref, "
        f"ki taper center={physics['ki_taper_center_eV']:.2f} eV, "
        f"width={physics['ki_taper_width_eV']:.2f} eV, "
        f"energy_alpha={physics['energy_alpha']:.2f}.",
        flush=True,
    )
    print(
        f"Discharge-current target: Id={physics['JT_target_A']:.3f} A, "
        f"A_channel={A_channel:.6e} m^2, JT_norm={physics['JT_norm']:.8g}, "
        f"JT_A_scale={physics['JT_A_scale']:.6g} A per normalized JT.",
        flush=True,
    )
    create_output_folders(OUTPUT_DIR)

    # 1b) Hybrid data loss (Haas & Gallimore P5 measurements).
    data_cfg = {"enabled": True, "weight_ni": 0.0, "weight_Te": 0.0, "weight_Vp": 0.0}
    cfg_path = os.path.join(_here, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                data_cfg = json.load(f).get("data_loss", {}) or {}
        except Exception as e:
            print(f"WARN: failed to parse config.json data_loss section: {e}",
                  flush=True)
            data_cfg = {}
    data_enabled = bool(data_cfg.get("enabled", True))
    exp_data = None
    exp_data_all = None
    w_data = {"ni": 1.0, "Te": 1.0, "Vp": 1.0}
    if data_enabled:
        w_data = {
            "ni": float(data_cfg.get("weight_ni", 1.0)),
            "Te": float(data_cfg.get("weight_Te", 1.0)),
            "Vp": float(data_cfg.get("weight_Vp", 1.0)),
        }
        csv_path = data_cfg.get("csv_path", "haas_gallimore_p5_5p4A.csv")
        if not os.path.isabs(csv_path):
            for base in (_here, _parent, os.getcwd()):
                cand = os.path.normpath(os.path.join(base, csv_path))
                if os.path.exists(cand):
                    csv_path = cand
                    break
        if not os.path.exists(csv_path):
            print(f"WARN: data_loss.csv_path not found ({csv_path}); "
                  "disabling data loss.", flush=True)
            data_enabled = False
        else:
            print(f"Hybrid data loss ENABLED (weights ni={w_data['ni']}, "
                  f"Te={w_data['Te']}, Vp={w_data['Vp']}); csv={csv_path}",
                  flush=True)
            exp_data_all = load_haas_gallimore_data(
                csv_path, domain, physics, device, dtype=torch.float64,
            )
            if CENTERLINE_DATA_R_MM is None:
                exp_data = exp_data_all
            else:
                print(
                    f"Training data loss restricted to r_mm={CENTERLINE_DATA_R_MM:g}; "
                    "other Haas/Gallimore radii are held out for evaluation only.",
                    flush=True,
                )
                exp_data = load_haas_gallimore_data(
                    csv_path, domain, physics, device, dtype=torch.float64,
                    include_r_mm=CENTERLINE_DATA_R_MM,
                )
    else:
        print("Hybrid data loss DISABLED (config.data_loss.enabled=false "
              "or section missing).", flush=True)
    base_w_data = dict(w_data)
    data_weight_scale = 1.0
    data_anneal_schedule = [
        {"start_epoch": 0, "end_epoch": EPOCHS_TO_RUN, "scale": 1.0},
    ]
    data_anneal_events = []

    def set_data_weight_scale(scale, epoch, reason):
        nonlocal data_weight_scale, w_data
        data_weight_scale = scale
        w_data = {k: base_w_data[k] * data_weight_scale for k in base_w_data}
        event = {
            "epoch": int(epoch),
            "scale": float(data_weight_scale),
            "weights": {k: float(v) for k, v in w_data.items()},
            "reason": reason,
        }
        data_anneal_events.append(event)
        print(
            f"Data-weight anneal @ epoch {epoch}: scale={data_weight_scale:.3g}, "
            f"weights ni/Te/Vp={w_data['ni']:.6g}/{w_data['Te']:.6g}/{w_data['Vp']:.6g} "
            f"({reason})",
            flush=True,
        )

    def annotate_data_weights(snap):
        snap["data_weight_scale"] = float(data_weight_scale)
        snap["data_weight_ni"] = float(w_data.get("ni", 0.0))
        snap["data_weight_Te"] = float(w_data.get("Te", 0.0))
        snap["data_weight_Vp"] = float(w_data.get("Vp", 0.0))

    if data_enabled and exp_data is not None:
        print(
            f"Data weights fixed for this run: ni/Te/Vp="
            f"{w_data['ni']:.6g}/{w_data['Te']:.6g}/{w_data['Vp']:.6g}; no annealing.",
            flush=True,
        )

    # 2) Build model. The seed above fixes init for future runs; if CKPT_IN is
    # provided, checkpoint weights immediately replace the initialized weights.
    model = build_model(domain, physics, device)
    ckpt_in_data = None
    if CKPT_IN:
        ckpt_in_data = torch.load(CKPT_IN, map_location=device)
        model.load_state_dict(ckpt_in_data["model_state_dict"], strict=True)
        print(
            f"Loaded checkpoint weights from {CKPT_IN}; "
            f"checkpoint epochs_completed={ckpt_in_data.get('epochs_completed', 'unknown')}.",
            flush=True,
        )
    physics["Vix_anode_min"] = float(getattr(model, "Vix_anode_min", physics["Vix_inlet"]))
    physics["ni_seed_peak"] = float(getattr(model, "ni_seed_peak", float("nan")))
    physics["ni_seed_amp"] = float(getattr(model, "ni_seed_amp", float("nan")))
    print("Output index mapping: 0=phi, 1=ni, 2=Vix, 3=Vir, 4=nn, "
          "5=ne, 6=Vex, 7=Ver, 8=Te, 9=Vetheta, 10=eta", flush=True)
    print(
        "Network raw outputs: 0=phi, 1=direct ni log-amplitude, 2=Vex correction, "
        "3=Vir, 4=nn depletion, 5=Ver, 6=Vetheta, 7=eta, 8=Te, 9=Vix.",
        flush=True,
    )
    print(
        "Phi form: phi = phi_L + (phi_R - phi_L)*xi "
        "+ xi*(1-xi)*25*tanh(raw0); end Dirichlet BCs are hard.",
        flush=True,
    )
    report_phi_bc_init(model, domain, device)
    report_phi_shape_init(model, domain, physics, device)
    report_ni_centerline_init(model, domain, device)
    report_ion_friction_init(model, domain, physics, device)
    print(f"Model initialized. eta_max={float(model.eta_max):.2e}, "
          f"ki_scale={physics['ki_scale']}", flush=True)

    # 3) Reproduce the EXACT same collocation and boundary points.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_f, r_f = sample_domain_points(N_DOMAIN, domain, device)
    boundary_data = sample_boundary_points(N_INLET, N_WALL, domain, device)

    # 4) Conservative loss weights: Adam only, electron momentum on at 10 for transported Vex.
    weights = {
        "poisson": 100.0,
        "ion_cont": 200.0,
        "neut_cont": 100.0,
        "elec_cont": 10.0,
        "energy": 1.0,
        "ion_mom_x": 20.0,
        "ion_mom_r": 20.0,
        "ni_inlet": 0.0,
        "nn_inlet": 200.0,
        "Vetheta_inlet": 20.0,
        "Vir_wall": 20.0,
        "Ver_wall": 10.0,
        "integrated_current": 100.0,
        "integrated_neutral": 100.0,
        "integrated_ion": 100.0,
        "nn_exit": 100.0,
        "smooth_ni": 1.0e-2,
        "smooth_nn": 1.0e-2,
        "smooth_phi": 1.0e-1,
        "smooth_Er": 0.0,
        "smooth_Te": 1.0e-3,
        "elec_mom_x": 10.0,
        "elec_mom_r": 10.0,
        "elec_mom_theta": 10.0,
    }
    if SMOOTH_NI_OVERRIDE not in (None, ""):
        weights["smooth_ni"] = float(SMOOTH_NI_OVERRIDE)
        print(f"smooth_ni overridden via env: {weights['smooth_ni']:.6g}", flush=True)

    # Calibrate residuals to O(1) only for fresh runs. Resume runs must restore
    # the normalization scales saved in the checkpoint; otherwise the same
    # weights can report a different loss immediately after loading.
    _, scale_probe = compute_all_losses(model, x_f, r_f, boundary_data, domain, physics, weights)
    restored_scale_keys = []
    if CKPT_IN and ckpt_in_data is not None:
        ckpt_physics = ckpt_in_data.get("physics", {}) or {}
        for scale_key in ("ion_cont_scale", "energy_scale", "ion_mom_scale"):
            if scale_key in ckpt_physics:
                physics[scale_key] = float(ckpt_physics[scale_key])
                restored_scale_keys.append(scale_key)
    if len(restored_scale_keys) == 3:
        print(
            "Restored residual scale factors from checkpoint: "
            f"ion_cont={physics['ion_cont_scale']:.6e}, "
            f"energy={physics['energy_scale']:.6e}, "
            f"ion_mom={physics['ion_mom_scale']:.6e}.",
            flush=True,
        )
    else:
        ion_cont_rms = math.sqrt(max(scale_probe.get("ion_cont_raw", scale_probe.get("ion_cont", 1.0)), 1e-12))
        energy_rms = math.sqrt(max(scale_probe.get("energy", 1.0), 1e-12))
        ion_mom_rms = math.sqrt(max(
            scale_probe.get("ion_mom_x", 1.0) + scale_probe.get("ion_mom_r", 0.0),
            1e-12,
        ))
        physics["ion_cont_scale"] = ion_cont_rms
        physics["energy_scale"] = energy_rms
        physics["ion_mom_scale"] = ion_mom_rms
        print(f"Ion continuity residual scale factor: {ion_cont_rms:.6e}", flush=True)
        print(f"Electron energy residual scale factor: {energy_rms:.6e}", flush=True)
        print(f"Ion momentum residual scale factor: {ion_mom_rms:.6e}", flush=True)
    physics["ion_cont_scale"] = 0.6100892858913114
    physics["energy_scale"] = 0.965239365788521
    physics["ion_mom_scale"] = 0.7936910647527818
    print(
        "Fixed reference residual scales: "
        f"ion_cont={physics['ion_cont_scale']:.16g}, "
        f"energy={physics['energy_scale']:.16g}, "
        f"ion_mom={physics['ion_mom_scale']:.16g}.",
        flush=True,
    )
    print(
        "Ion force sign check: old 1D closure gives Vix*dVix/dx + dphi/dx = 0; "
        "therefore conservative residual uses +ni*dphi/dx.",
        flush=True,
    )
    print(f"Typical collision magnitudes at init: nu_ei={scale_probe.get('nu_ei_mean_Hz', float('nan')):.3e} Hz, "
          f"nu_en={scale_probe.get('nu_en_mean_Hz', float('nan')):.3e} Hz", flush=True)

    # 5) Optimizer. Adam is the established baseline; SOAP is enabled only by
    # explicit env override and uses the upstream recommended defaults.
    optimizer_hyperparams = {}
    if OPTIMIZER_OVERRIDE == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        optimizer_hyperparams = {"lr": float(LR)}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=300, min_lr=1e-6
        )
    elif OPTIMIZER_OVERRIDE == "SOAP":
        from soap import SOAP
        soap_kwargs = {
            "lr": 3.0e-3,
            "betas": (0.95, 0.95),
            "weight_decay": 0.01,
            "precondition_frequency": 10,
        }
        optimizer = SOAP(model.parameters(), **soap_kwargs)
        optimizer_hyperparams = {
            "lr": soap_kwargs["lr"],
            "betas": list(soap_kwargs["betas"]),
            "weight_decay": soap_kwargs["weight_decay"],
            "precondition_frequency": soap_kwargs["precondition_frequency"],
            "source": "nikhilvyas/SOAP soap.py defaults/recommended README example",
        }
        scheduler = None
        print(f"SOAP optimizer enabled with hyperparameters: {optimizer_hyperparams}", flush=True)
    else:
        raise ValueError(f"Unsupported OPTIMIZER_OVERRIDE={OPTIMIZER_OVERRIDE!r}")

    history = init_history()
    # Extra history keys for hybrid data loss (always present; zero when disabled).
    for _k in (
        "L_data_ni", "L_data_Te", "L_data_Vp", "L_data_total",
        "energy", "ion_mom_r", "elec_mom_x", "elec_mom_r", "elec_mom_theta",
        "integrated_neutral", "integrated_ion", "Id_mean_A", "smooth_phi", "smooth_Er", "smooth_Te",
    ):
        history[_k] = []
    stop_reason = f"completed {EPOCHS_START + EPOCHS_TO_RUN} {OPTIMIZER_OVERRIDE} epochs + {LBFGS_STEPS} L-BFGS steps"
    final_epoch = EPOCHS_START

    def add_hybrid_data_loss(total, loss_dict):
        if data_enabled and exp_data is not None:
            data_losses = compute_data_loss(model, exp_data)
            weighted_data = (
                w_data["ni"] * data_losses["ni"]
                + w_data["Te"] * data_losses["Te"]
                + w_data["Vp"] * data_losses["Vp"]
            )
            total = total + weighted_data
            loss_dict["L_data_ni"] = float(data_losses["ni"].detach().item())
            loss_dict["L_data_Te"] = float(data_losses["Te"].detach().item())
            loss_dict["L_data_Vp"] = float(data_losses["Vp"].detach().item())
            loss_dict["L_data_total"] = float(weighted_data.detach().item())
            loss_dict["total"] = float(total.detach().item())
        else:
            loss_dict["L_data_ni"] = 0.0
            loss_dict["L_data_Te"] = 0.0
            loss_dict["L_data_Vp"] = 0.0
            loss_dict["L_data_total"] = 0.0
        return total, loss_dict

    # 6) Capture starting snapshot for the trajectory table.
    trajectory = []
    start_snap = _evaluate_monitor(model, domain, physics, device)
    start_snap["epoch"] = EPOCHS_START
    init_total, init_loss_dict = compute_all_losses(model, x_f, r_f, boundary_data, domain, physics, weights)
    init_total, init_loss_dict = add_hybrid_data_loss(init_total, init_loss_dict)
    start_snap["total_loss"] = float(init_loss_dict["total"])
    annotate_data_weights(start_snap)
    loss_explosion_limit = max(LOSS_EXPLOSION_LIMIT, 10.0 * float(init_loss_dict["total"]))
    data_initial = None
    if data_enabled and exp_data is not None:
        init_data_losses = compute_data_loss(model, exp_data)
        data_initial = {
            "losses": {k: float(v.detach().item()) for k, v in init_data_losses.items()},
            "l2": data_loss_l2_relative(model, exp_data),
        }
    trajectory.append(start_snap)
    print(
        f"[epoch {EPOCHS_START:5d}] total={start_snap['total_loss']:.6f}  "
        f"ni_peak={start_snap['ni_peak_value']:.4f}@x={start_snap['ni_peak_x']:.3f}  "
        f"sat95={start_snap['sat_frac_95']:.3f}  "
        f"IonCont={start_snap['IonCont_RMS']:.4f} "
        f"NeutCont={start_snap['NeutCont_RMS']:.4f} "
        f"Poisson={start_snap['Poisson_RMS']:.4f} "
        f"wdata={data_weight_scale:.3g}",
        flush=True,
    )

    # 7) Continuation loop.
    t0 = time.time()
    early_stop = False
    last_loss_dict = dict(init_loss_dict)

    def weighted_loss_breakdown(loss_dict):
        terms = {}
        for key, weight_key in (
            ("poisson", "poisson"),
            ("ion_cont", "ion_cont"),
            ("neut_cont", "neut_cont"),
            ("elec_cont", "elec_cont"),
            ("energy", "energy"),
            ("ion_mom_x", "ion_mom_x"),
            ("ion_mom_r", "ion_mom_r"),
            ("elec_mom_x", "elec_mom_x"),
            ("elec_mom_r", "elec_mom_r"),
            ("elec_mom_theta", "elec_mom_theta"),
            ("ni_inlet", "ni_inlet"),
            ("nn_inlet", "nn_inlet"),
            ("nn_exit", "nn_exit"),
            ("Vetheta_inlet", "Vetheta_inlet"),
            ("Vir_wall", "Vir_wall"),
            ("Ver_wall", "Ver_wall"),
            ("integrated_current", "integrated_current"),
            ("integrated_neutral", "integrated_neutral"),
            ("integrated_ion", "integrated_ion"),
            ("smooth_ni", "smooth_ni"),
            ("smooth_nn", "smooth_nn"),
            ("smooth_phi", "smooth_phi"),
            ("smooth_Er", "smooth_Er"),
            ("smooth_Te", "smooth_Te"),
        ):
            terms[key] = float(weights.get(weight_key, 0.0) * loss_dict.get(key, 0.0))
        terms["data_ni"] = float(w_data.get("ni", 0.0) * loss_dict.get("L_data_ni", 0.0))
        terms["data_Te"] = float(w_data.get("Te", 0.0) * loss_dict.get("L_data_Te", 0.0))
        terms["data_Vp"] = float(w_data.get("Vp", 0.0) * loss_dict.get("L_data_Vp", 0.0))
        terms["data_total"] = float(loss_dict.get("L_data_total", 0.0))
        terms["total_from_terms"] = float(
            sum(v for k, v in terms.items() if k not in ("data_total", "total_from_terms"))
        )
        terms["reported_total"] = float(loss_dict.get("total", float("nan")))
        return terms

    def annotate_weighted_losses(snap, loss_dict):
        terms = weighted_loss_breakdown(loss_dict)
        snap["weighted_loss_terms"] = terms
        snap["weighted_loss_terms_sorted"] = [
            {"term": k, "weighted_value": v}
            for k, v in sorted(
                ((k, v) for k, v in terms.items() if k not in ("total_from_terms", "reported_total")),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]

    def append_history(loss_dict):
        for hk in history:
            if hk not in ("ni_mean", "ni_std") and hk in loss_dict:
                history[hk].append(loss_dict[hk])
        history["ni_mean"].append(
            history["ni_mean"][-1] if history["ni_mean"] else 0.0
        )
        history["ni_std"].append(
            history["ni_std"][-1] if history["ni_std"] else 0.0
        )

    for k in range(EPOCHS_TO_RUN):
        epoch = EPOCHS_START + k + 1
        optimizer.zero_grad()
        total, loss_dict = compute_all_losses(
            model, x_f, r_f, boundary_data, domain, physics, weights
        )
        # Hybrid data loss: added to total before backward; unweighted MSEs are
        # logged into loss_dict (and therefore into history) for diagnostics.
        total, loss_dict = add_hybrid_data_loss(total, loss_dict)
        last_loss_dict = dict(loss_dict)
        if not torch.isfinite(total):
            stop_reason = f"NaN/Inf loss at epoch {epoch}"
            early_stop = True
            final_epoch = epoch
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss_dict["total"])

        append_history(loss_dict)

        final_epoch = epoch

        if epoch % PRINT_EVERY == 0 and epoch % MONITOR_EVERY != 0:
            with torch.no_grad():
                _, ni_c, _, _, _, _, _, _, _, _, _ = model(x_f.detach(), r_f.detach())
                xi_check = (x_f.detach() - domain["x_min"]) / (
                    domain["x_max"] - domain["x_min"]
                )
                interior_mask = xi_check > 0.05
                ni_int_max = (
                    ni_c[interior_mask].max().item()
                    if interior_mask.any() else ni_c.mean().item()
                )
            data_str = (
                f" | Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                if data_enabled else ""
            )
            print(
                f"  epoch {epoch:5d} | total={loss_dict['total']:.4f} | "
                f"Poisson={loss_dict['poisson']:.3e} | "
                f"IonCont={loss_dict['ion_cont']:.4f} | "
                f"NeutCont={loss_dict['neut_cont']:.4f} | "
                f"|eta|={loss_dict['eta_abs_mean']:.2e} | "
                f"Id_mean_A={loss_dict['Id_mean_A']:.4f} | "
                f"ni int_max={ni_int_max:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"wdata={data_weight_scale:.3g}"
                + data_str,
                flush=True,
            )

        if epoch % MONITOR_EVERY == 0:
            snap = _evaluate_monitor(model, domain, physics, device)
            snap["epoch"] = epoch
            snap["total_loss"] = float(loss_dict["total"])
            annotate_data_weights(snap)
            annotate_weighted_losses(snap, loss_dict)
            trajectory.append(snap)
            data_str = (
                f"  Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                if data_enabled else ""
            )
            print(
                f"[epoch {epoch:5d}] total={snap['total_loss']:.6f}  "
                f"ni_peak={snap['ni_peak_value']:.4f}@x={snap['ni_peak_x']:.3f}  "
                f"sat95={snap['sat_frac_95']:.3f}  "
                f"IonCont={snap['IonCont_RMS']:.4f} "
                f"NeutCont={snap['NeutCont_RMS']:.4f} "
                f"Poisson={snap['Poisson_RMS']:.4f}"
                f"  wdata={data_weight_scale:.3g}"
                + data_str,
                flush=True,
            )
            # Apply hard stops only at monitor checkpoints (per user spec).
            if snap["sat_frac_95"] > STOP_ETA_SAT_95:
                stop_reason = (
                    f"eta-saturation stop at epoch {epoch}: "
                    f"sat_frac_95={snap['sat_frac_95']:.4f} > {STOP_ETA_SAT_95}"
                )
                early_stop = True
                break
            if snap["ni_peak_value"] < STOP_NI_PEAK_MIN:
                stop_reason = (
                    f"ni-peak collapse at epoch {epoch}: "
                    f"ni_peak={snap['ni_peak_value']:.4f} < {STOP_NI_PEAK_MIN}"
                )
                early_stop = True
                break

        # Loss-explosion guard, same threshold as the original run.
        # When the hybrid data loss is enabled, compare against the PDE-only
        # subtotal — the explosion threshold was calibrated against v24's PDE
        # losses, so a large data MSE at warm-start must not trip it.
        pde_total = loss_dict["total"] - loss_dict.get("L_data_total", 0.0)
        if pde_total > loss_explosion_limit:
            stop_reason = (
                f"loss explosion at epoch {epoch}: "
                f"pde={pde_total:.6g} > {loss_explosion_limit:.6g}"
            )
            early_stop = True
            break

    if (not early_stop) and LBFGS_STEPS > 0:
        print(f"Starting L-BFGS refinement for {LBFGS_STEPS} steps.", flush=True)
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=1,
            max_eval=5,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        lbfgs_last = {"loss": None, "loss_dict": dict(last_loss_dict)}

        for j in range(LBFGS_STEPS):
            epoch = EPOCHS_START + EPOCHS_TO_RUN + j + 1

            def closure():
                lbfgs_optimizer.zero_grad()
                total_c, loss_dict_c = compute_all_losses(
                    model, x_f, r_f, boundary_data, domain, physics, weights
                )
                total_c, loss_dict_c = add_hybrid_data_loss(total_c, loss_dict_c)
                if torch.isfinite(total_c):
                    total_c.backward()
                lbfgs_last["loss"] = total_c.detach()
                lbfgs_last["loss_dict"] = dict(loss_dict_c)
                return total_c

            lbfgs_optimizer.step(closure)
            loss_dict = dict(lbfgs_last["loss_dict"])
            total_value = float(loss_dict.get("total", float("nan")))
            last_loss_dict = dict(loss_dict)
            append_history(loss_dict)
            final_epoch = epoch

            if (not math.isfinite(total_value)):
                stop_reason = f"NaN/Inf loss during L-BFGS at step {j + 1}"
                early_stop = True
                break

            if epoch % PRINT_EVERY == 0 and epoch % MONITOR_EVERY != 0:
                with torch.no_grad():
                    _, ni_c, _, _, _, _, _, _, _, _, _ = model(x_f.detach(), r_f.detach())
                    xi_check = (x_f.detach() - domain["x_min"]) / (
                        domain["x_max"] - domain["x_min"]
                    )
                    interior_mask = xi_check > 0.05
                    ni_int_max = (
                        ni_c[interior_mask].max().item()
                        if interior_mask.any() else ni_c.mean().item()
                    )
                data_str = (
                    f" | Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                    f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                    if data_enabled else ""
                )
                print(
                    f"  lbfgs {j + 1:5d} (epoch {epoch:5d}) | total={total_value:.4f} | "
                    f"Poisson={loss_dict['poisson']:.3e} | "
                    f"IonCont={loss_dict['ion_cont']:.4f} | "
                    f"NeutCont={loss_dict['neut_cont']:.4f} | "
                    f"|eta|={loss_dict['eta_abs_mean']:.2e} | "
                    f"Id_mean_A={loss_dict['Id_mean_A']:.4f} | "
                    f"ni int_max={ni_int_max:.4f}"
                    + data_str,
                    flush=True,
                )

            if epoch % MONITOR_EVERY == 0:
                snap = _evaluate_monitor(model, domain, physics, device)
                snap["epoch"] = epoch
                snap["total_loss"] = total_value
                annotate_data_weights(snap)
                annotate_weighted_losses(snap, loss_dict)
                trajectory.append(snap)
                data_str = (
                    f"  Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                    f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                    if data_enabled else ""
                )
                print(
                    f"[lbfgs {j + 1:5d} | epoch {epoch:5d}] total={total_value:.6f}  "
                    f"ni_peak={snap['ni_peak_value']:.4f}@x={snap['ni_peak_x']:.3f}  "
                    f"sat95={snap['sat_frac_95']:.3f}  "
                    f"IonCont={snap['IonCont_RMS']:.4f} "
                    f"NeutCont={snap['NeutCont_RMS']:.4f} "
                    f"Poisson={snap['Poisson_RMS']:.4f}"
                    + data_str,
                    flush=True,
                )
                if snap["sat_frac_95"] > STOP_ETA_SAT_95:
                    stop_reason = (
                        f"eta-saturation stop during L-BFGS step {j + 1}: "
                        f"sat_frac_95={snap['sat_frac_95']:.4f} > {STOP_ETA_SAT_95}"
                    )
                    early_stop = True
                    break
                if snap["ni_peak_value"] < STOP_NI_PEAK_MIN:
                    stop_reason = (
                        f"ni-peak collapse during L-BFGS step {j + 1}: "
                        f"ni_peak={snap['ni_peak_value']:.4f} < {STOP_NI_PEAK_MIN}"
                    )
                    early_stop = True
                    break

            pde_total = loss_dict["total"] - loss_dict.get("L_data_total", 0.0)
            if pde_total > loss_explosion_limit:
                stop_reason = (
                    f"loss explosion during L-BFGS step {j + 1}: "
                    f"pde={pde_total:.6g} > {loss_explosion_limit:.6g}"
                )
                early_stop = True
                break

    elapsed = time.time() - t0
    print(f"Training stopped: {stop_reason}", flush=True)
    print(f"Elapsed: {elapsed/60.0:.2f} min", flush=True)

    # 8) Save final checkpoint with trajectory metadata.
    model.eval()
    physics_out = dict(physics)
    physics_out["Id_mean_A"] = mean_current_A(model, domain, physics)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "physics": physics_out,
            "domain": domain,
            "eta_max": float(model.eta_max),
            "run_name": RUN_NAME,
            "epochs_completed": final_epoch,
            "init_seed": INIT_SEED,
            "collocation_seed": SEED,
            "weights": weights,
            "adam_epochs_requested": EPOCHS_TO_RUN if OPTIMIZER_OVERRIDE == "ADAM" else 0,
            "optimizer_name": OPTIMIZER_OVERRIDE,
            "optimizer_steps_requested": EPOCHS_TO_RUN,
            "optimizer_hyperparams": optimizer_hyperparams,
            "lbfgs_steps_requested": LBFGS_STEPS,
            "stop_reason": stop_reason,
            "final_loss": history["total"][-1] if history["total"] else float("nan"),
            "trajectory": trajectory,
            "data_anneal": {
                "scheme": "fixed_full_data_no_annealing",
                "schedule": data_anneal_schedule,
                "base_weights": {k: float(v) for k, v in base_w_data.items()},
                "events": data_anneal_events,
                "final_scale": float(data_weight_scale),
                "final_weights": {k: float(v) for k, v in w_data.items()},
                "centerline_data_r_mm": CENTERLINE_DATA_R_MM,
            },
            "resume_from": CKPT_IN,
        },
        CKPT_OUT,
    )
    print(f"Checkpoint saved: {CKPT_OUT}", flush=True)

    # The isolated runner evaluates the new closure and writes the comparison report.
    return

if __name__ == "__main__":
    main()
