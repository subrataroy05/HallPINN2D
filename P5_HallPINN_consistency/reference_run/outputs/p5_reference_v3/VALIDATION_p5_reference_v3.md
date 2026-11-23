# V3 validation report

Run name: p5_physics_only
Training history: completed 15000 ADAM epochs + 0 L-BFGS steps. Final optimizer epoch/step index = 15000.
Requested schedule: Adam epochs = 15000; L-BFGS steps = 0.
Final total loss = 7.33974. Early stop = False.
Loss explosion guard used = 13843.3.

## Output index mapping

Forward returns: 0=phi, 1=ni, 2=Vix, 3=Vir, 4=nn, 5=ne, 6=Vex, 7=Ver, 8=Te, 9=Vetheta, 10=eta.
Network raw outputs: 0=phi correction, 1=direct ni log-amplitude, 2=Vex correction, 3=Vir, 4=nn depletion, 5=Ver, 6=Vetheta, 7=eta, 8=Te, 9=Vix.

## Normalization and constants

- phi_ref = 12.1 eV.
- nu_ref = 1.761146e+06 1/s.
- tau_ref = L_canal/V_ref = 1.274348e-05 s.
- ion continuity residual scale = 6.100893e-01.
- electron energy residual scale = 9.652394e-01.
- ion momentum residual scale = 7.936911e-01.
- ion-neutral collision frequency nu_in = 1.800000e+06 s^-1; nu_in_hat = nu_in*tau_ref = 22.9383.
- discharge-current target = 5.4 A; A_channel = 1.138827e-02 m^2; JT_norm = 0.02959875; JT_A_scale = 182.44 A per normalized JT.
- explicit current penalty: L_current = (JT_hat*JT_A_scale - 5.4 A)^2, weight = 5.
- ki low-Te taper = sigmoid((Te - 5 eV) / 0.35 eV).
- Ion force sign check: old 1D closure gives Vix*dVix/dx + dphi/dx = 0, so the conservative residual uses +ni*dphi/dx.
- nu_ei final mean = 3.193593e+05 Hz.
- nu_en final mean = 4.606898e+06 Hz.
- final mean |ion friction/loading/e-drag x| = 3.383249e-01; mean |ni*dphi/dx| = 3.590803e-01; friction/electric = 0.942198.
- Hall momentum weights: x=0.01, r=0.01, theta=0.01.
- Data weights: ni=10, Te=5, Vp=30.
- Data ni loss = log-density MSE; Te and Vp data losses = normalized MSE.
- Ion continuity weight = 200.
- Ion momentum weights: x=20, r=20.
- Direct-ni hard anode target = 0.0138361 normalized = 4.640000e+17 m^-3.
- Direct-ni HG-amplitude seed peak = 0.06 normalized; seed amp = 1.46706.
- Vix relaxed anode target = 0.025 normalized = 74.5479 m/s (old Bohm target 1490.96 m/s).
- Neutral axial bulk speed Vn = 450 m/s (0.15091 normalized).

Guessed constants flagged for review:

- energy_alpha = 2.5 (excitation/ionization multiplier).
- sigma_en = 2.500e-19 m^2 for Xe.
- Coulomb log = 10.
- Tn = 0.025 eV and Ti = 0.025 eV.

## Before/after losses

| Term | init | final |
|---|---:|---:|
| total | 1384.33 | 7.33974 |
| poisson | 5.36295e-06 | 7.60461e-05 |
| ion_cont | 1 | 0.00338366 |
| ion_cont_raw | 0.372209 | 0.00125943 |
| neut_cont | 0.279176 | 0.0224476 |
| elec_cont | 0.372165 | 0.0012626 |
| energy | 5.52152 | 0.121311 |
| ion_mom_x | 10.7942 | 0.0179866 |
| ion_mom_r | 0.00196434 | 1.01125e-05 |
| elec_mom_x | 0.413401 | 0.293192 |
| elec_mom_r | 0.00113054 | 5.72295e-07 |
| elec_mom_theta | 0.000198784 | 2.26778e-06 |
| JT_target | 0 | 4.10246e-12 |
| JT_current_A | 0 | 1.36548e-07 |
| nn_exit | 0 | 0 |
| smooth_ni | 0 | 1.24114 |
| smooth_nn | 1.58124e-05 | 1.31266 |
| smooth_phi | 0.121945 | 0.0195984 |
| smooth_Er | 0.121945 | 0.0195984 |
| smooth_Te | 0.000138197 | 29.8528 |

## Field metrics

- ni_peak_value = 0.076737284; ni_peak_x = 0.50753769.
- Te_peak_value = 3.086874 normalized = 37.351175 eV; Te_peak_x = 0.5879397.
- Ex_peak_x = 1.
- nn(x=0.55, centerline) = 0.0067670913.
- RMS (200x80): IonCont raw = 0.035360189; IonCont scaled = nan; NeutCont = 0.15031138; Poisson = 0.0087016452.
- eta min/max/mean = -9.9676099e-05 / 9.6639924e-05 / -2.850319e-05.
- eta sat95 fraction = 0.140125.
- JT residual RMS = 1.45215110e-18.
- JT trained = 0.029596818 normalized = 5.3996475 A (target 5.4 A).

## Data loss

| Quantity | init MSE | final MSE | init L2 rel | final L2 rel |
|---|---:|---:|---:|---:|
| ni | 0.892467 | 0.0340426 | 0.667802 | 0.200811 |
| Te | 2.30411 | 0.0217631 | 0.721647 | 0.0701243 |
| Vp | 30.3594 | 0.113569 | 0.282731 | 0.0172942 |

## Training trajectory

| epoch | total_loss | ni_peak | ni_peak_x | sat_frac_95 | IonCont_RMS | NeutCont_RMS | Poisson_RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1384.3 | 0.059998 | 0.5025 | 0 | 0.61465 | 0.52646 | 0.0023103 |
| 500 | 27.259 | 0.034497 | 0.4322 | 0 | 0.028876 | 0.021135 | 0.0049307 |
| 1000 | 25.637 | 0.027547 | 0.4422 | 0 | 0.02601 | 0.023862 | 0.0045624 |
| 1500 | 24.99 | 0.028005 | 0.6533 | 0 | 0.029051 | 0.033791 | 0.0036165 |
| 2000 | 13.332 | 0.033613 | 0.5528 | 0 | 0.044962 | 0.1581 | 0.0047642 |
| 2500 | 12.275 | 0.038047 | 0.5528 | 0 | 0.041111 | 0.15343 | 0.00593 |
| 3000 | 11.111 | 0.041052 | 0.5578 | 0 | 0.040831 | 0.15191 | 0.0067135 |
| 3500 | 9.7565 | 0.045535 | 0.5578 | 0 | 0.039967 | 0.15142 | 0.0070294 |
| 4000 | 8.3039 | 0.057436 | 0.5678 | 0 | 0.038822 | 0.15264 | 0.0075279 |
| 4500 | 7.9496 | 0.066532 | 0.5678 | 0 | 0.037438 | 0.15335 | 0.0076102 |
| 5000 | 7.8028 | 0.070475 | 0.5628 | 0 | 0.036878 | 0.15346 | 0.0075656 |
| 5500 | 7.7062 | 0.072419 | 0.5628 | 0 | 0.03672 | 0.15327 | 0.0075054 |
| 6000 | 7.637 | 0.073803 | 0.5578 | 0 | 0.036696 | 0.15284 | 0.0075393 |
| 6500 | 7.5823 | 0.074682 | 0.5528 | 0 | 0.036709 | 0.15228 | 0.0075618 |
| 7000 | 7.5346 | 0.075291 | 0.5427 | 0 | 0.036787 | 0.15156 | 0.0076325 |
| 7500 | 7.4988 | 0.075624 | 0.5377 | 0 | 0.036747 | 0.15094 | 0.007783 |
| 8000 | 7.4682 | 0.07588 | 0.5377 | 0 | 0.036536 | 0.1507 | 0.007866 |
| 8500 | 7.4414 | 0.076158 | 0.5327 | 0 | 0.036349 | 0.15057 | 0.0079201 |
| 9000 | 7.4213 | 0.076423 | 0.5276 | 0.009875 | 0.036123 | 0.15052 | 0.0079896 |
| 9500 | 7.4057 | 0.07657 | 0.5226 | 0.02825 | 0.035975 | 0.15047 | 0.0080585 |
| 10000 | 7.3949 | 0.076779 | 0.5226 | 0.04206 | 0.035856 | 0.1504 | 0.0081543 |
| 10500 | 7.3866 | 0.076861 | 0.5176 | 0.05519 | 0.035795 | 0.15027 | 0.0082519 |
| 11000 | 7.3784 | 0.076972 | 0.5176 | 0.0685 | 0.035652 | 0.15028 | 0.0083587 |
| 11500 | 7.3711 | 0.077006 | 0.5126 | 0.08069 | 0.03559 | 0.15024 | 0.0084553 |
| 12000 | 7.3636 | 0.07703 | 0.5126 | 0.09175 | 0.035535 | 0.15024 | 0.0085287 |
| 12500 | 7.3602 | 0.07716 | 0.5126 | 0.1012 | 0.035373 | 0.15033 | 0.0086199 |
| 13000 | 7.3553 | 0.077101 | 0.5075 | 0.1086 | 0.035356 | 0.15032 | 0.0086582 |
| 13500 | 7.3507 | 0.077067 | 0.5075 | 0.1154 | 0.035323 | 0.15035 | 0.0086792 |
| 14000 | 7.3456 | 0.0769 | 0.5075 | 0.1241 | 0.035413 | 0.15027 | 0.0086815 |
| 14500 | 7.3431 | 0.076982 | 0.5075 | 0.1324 | 0.035217 | 0.15047 | 0.0087145 |
| 15000 | 7.3397 | 0.07679 | 0.5075 | 0.1401 | 0.03536 | 0.15031 | 0.0087016 |

## Stop diagnosis

Completed the requested validation epochs without triggering an early-stop rule.
