import argparse
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import torch


WEIGHTS = {
    "poisson": 100.0, "ion_cont": 200.0, "neut_cont": 100.0,
    "elec_cont": 10.0, "energy": 1.0, "ion_mom_x": 20.0,
    "ion_mom_r": 20.0, "elec_mom_x": 0.01, "elec_mom_r": 0.01,
    "elec_mom_theta": 0.01, "ni_inlet": 0.0, "nn_inlet": 200.0,
    "Vetheta_inlet": 20.0, "Vir_wall": 20.0, "Ver_wall": 10.0,
    "JT_target": 100.0, "JT_current_A": 5.0, "nn_exit": 100.0,
    "smooth_ni": 0.01, "smooth_nn": 0.01, "smooth_phi": 0.1,
    "smooth_Er": 0.0, "smooth_Te": 0.001,
    "data_ni": 10.0, "data_Te": 5.0, "data_Vp": 30.0,
}
EQUATIONS = [
    "poisson", "ion_cont", "neut_cont", "elec_cont", "ion_mom_x",
    "ion_mom_r", "elec_mom_x", "elec_mom_r", "elec_mom_theta", "energy",
    "current_closure",
]


def banding(field, r):
    dr = float(np.mean(np.diff(r)))
    d2 = (field[:, 2:] - 2 * field[:, 1:-1] + field[:, :-2]) / dr**2
    centered = field - field.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    total = float(power[:, 1:].sum())
    return {
        "d2r2_rms": float(np.sqrt(np.mean(d2**2))),
        "fft_kge4_fraction": float(power[:, 4:].sum()) / total,
    }


def stabilization(trajectory, term):
    points = []
    for snap in trajectory:
        terms = snap.get("weighted_loss_terms", {})
        if term in terms:
            points.append((int(snap["epoch"]), float(terms[term])))
    if not points:
        return {"first_within_5pct": None, "settled_within_5pct": None}
    final = points[-1][1]
    tolerance = 0.05 * abs(final)
    inside = [abs(value - final) <= tolerance for _, value in points]
    first = next((points[i][0] for i, ok in enumerate(inside) if ok), None)
    settled = next((points[i][0] for i in range(len(points)) if all(inside[i:])), None)
    return {"first_within_5pct": first, "settled_within_5pct": settled}


def residual_mse(equation):
    names = {
        "poisson": "poisson", "ion_cont": "ion_cont", "neut_cont": "neut_cont",
        "elec_cont": "elec_cont", "ion_mom_x": "ion_mom_x",
        "ion_mom_r": "ion_mom_r", "elec_mom_x": "elec_mom_x",
        "elec_mom_r": "elec_mom_r", "elec_mom_theta": "elec_mom_theta",
        "energy": "energy", "current_closure": "current",
    }
    return {out: float(equation[key].square().mean().item()) for out, key in names.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.code))
    from mainScript import build_model
    from utils import (
        compute_all_losses, compute_equation_residuals, data_loss_l2_relative,
        evaluate_on_grid, load_haas_gallimore_data, sample_boundary_points,
        sample_domain_points,
    )

    torch.set_default_dtype(torch.float64)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    physics, domain = ckpt["physics"], ckpt["domain"]
    device = torch.device("cpu")
    model = build_model(domain, physics, device).double()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    torch.manual_seed(2601)
    np.random.seed(2601)
    x_f, r_f = sample_domain_points(10000, domain, device)
    boundary = sample_boundary_points(1000, 1000, domain, device)
    final_collocation = residual_mse(compute_equation_residuals(model, x_f, r_f, physics))

    xg = torch.linspace(domain["x_min"], domain["x_max"], 200, dtype=torch.float64)
    rg = torch.linspace(domain["r_min"], domain["r_max"], 80, dtype=torch.float64)
    XXt, RRt = torch.meshgrid(xg, rg, indexing="ij")
    xf = XXt.reshape(-1).requires_grad_(True)
    rf = RRt.reshape(-1).requires_grad_(True)
    final_grid = residual_mse(compute_equation_residuals(model, xf, rf, physics))

    with contextlib.redirect_stdout(io.StringIO()):
        exp_data = load_haas_gallimore_data(args.csv, domain, physics, device, dtype=torch.float64)
    fit = data_loss_l2_relative(model, exp_data)
    values = evaluate_on_grid(model, domain, physics, nx=200, nr=80, device=device)
    XX, RR, phi, ni, Vix, Vir, nn, ne, Vex, Ver, Te, Vt, Si, Jx, ki, Ex, Er, eta = values

    torch.manual_seed(args.init_seed)
    np.random.seed(args.init_seed)
    initial_model = build_model(domain, physics, device).double()
    torch.manual_seed(2601)
    np.random.seed(2601)
    xi, ri = sample_domain_points(10000, domain, device)
    bi = sample_boundary_points(1000, 1000, domain, device)
    initial_residuals = residual_mse(compute_equation_residuals(initial_model, xi, ri, physics))
    initial_total, initial_loss = compute_all_losses(
        initial_model, xi, ri, bi, domain, physics, WEIGHTS
    )
    from utils import compute_data_loss
    initial_data = compute_data_loss(initial_model, exp_data)
    initial_total = initial_total + sum(
        WEIGHTS[f"data_{key}"] * initial_data[key] for key in ("ni", "Te", "Vp")
    )
    contributions = {key: WEIGHTS[key] * float(value) for key, value in initial_loss.items() if key in WEIGHTS}
    contributions.update({f"data_{key}": WEIGHTS[f"data_{key}"] * float(initial_data[key]) for key in ("ni", "Te", "Vp")})
    top3 = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:3]

    rmid = ni.shape[1] // 2
    ni_line, te_line = ni[:, rmid], Te[:, rmid]
    nii, tei = int(np.argmax(ni_line)), int(np.argmax(te_line))
    r_phys = RR[0] * physics["L_canal"]
    inlet = float(np.trapz(nn[0] * r_phys, r_phys) * 2 * np.pi * physics["Vn"])
    outlet = float(np.trapz(nn[-1] * r_phys, r_phys) * 2 * np.pi * physics["Vn"])
    source = float(np.trapz(np.trapz(Si * r_phys[None, :], r_phys, axis=1) * 2 * np.pi, XX[:, 0]))
    last = ckpt["trajectory"][-1]

    convergence_terms = [
        "poisson", "ion_cont", "neut_cont", "elec_cont", "ion_mom_x",
        "ion_mom_r", "elec_mom_x", "elec_mom_r", "elec_mom_theta", "energy",
        "JT_target", "data_ni", "data_Te", "data_Vp",
    ]
    result = {
        "checkpoint": args.checkpoint,
        "epochs_completed": ckpt["epochs_completed"], "stop_reason": ckpt["stop_reason"],
        "fit": fit, "JT_A": physics["JT_trained"] * physics["JT_A_scale"],
        "JT_percent_from_5p4": (physics["JT_trained"] * physics["JT_A_scale"] / 5.4 - 1) * 100,
        "grid_mse": final_grid, "grid_sum": sum(final_grid.values()),
        "collocation_mse": final_collocation, "collocation_sum": sum(final_collocation.values()),
        "initial_collocation_mse": initial_residuals,
        "theta_initial_collocation_mse": initial_residuals["elec_mom_theta"],
        "theta_final_grid_share": final_grid["elec_mom_theta"] / sum(final_grid.values()),
        "theta_final_collocation_share": final_collocation["elec_mom_theta"] / sum(final_collocation.values()),
        "Vetheta_max_abs_norm": float(np.max(np.abs(Vt))),
        "Vetheta_max_abs_m_s": float(np.max(np.abs(Vt)) * physics["V_ref"]),
        "Vetheta_bound_fraction": float(np.max(np.abs(Vt)) / 0.0125),
        "ni_peak": float(ni_line[nii]), "ni_peak_x": float(XX[nii, rmid]),
        "Te_peak_eV": float(te_line[tei] * physics["phi_ref"]), "Te_peak_x": float(XX[tei, rmid]),
        "Vix_exit_centerline_km_s": float(Vix[-1, rmid] * physics["V_ref"] / 1000),
        "eta_min": float(np.min(eta)), "eta_max": float(np.max(eta)),
        "eta_sat95": float(last["sat_frac_95"]),
        "weighted_loss_epoch0": float(initial_total.detach().item()),
        "weighted_loss_final": float(ckpt["final_loss"]), "epoch0_top3": top3,
        "convergence_5pct": {term: stabilization(ckpt["trajectory"], term) for term in convergence_terms},
        "depletion_to_source_ratio": (inlet - outlet) / source,
        "ion_cont_raw_rms": math.sqrt(float(compute_equation_residuals(model, xf, rf, physics)["ion_cont_raw"].square().mean().item())),
        "ion_cont_scaled_rms": math.sqrt(final_grid["ion_cont"]),
        "banding_ni": banding(ni, RR[0]), "banding_ne": banding(ne, RR[0]),
    }
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
