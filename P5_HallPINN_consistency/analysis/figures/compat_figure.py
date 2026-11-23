"""Constraint-compatibility map (figure version of Table 5), drawn from compat_map.csv values."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 7.2})

OK, MID, NO, NA = "ok", "mid", "no", "na"
FILL = {OK: "#dcefd8", MID: "#fbf0c6", NO: "#f6d5d2", NA: "#ffffff"}
MARK = {OK: "✓", MID: "△", NO: "✗", NA: ""}

# columns: n_i, T_e, V_p, ions, neutrals, current, I_DD, W_res/K
rows = [
    ("Reference (Gaussian $B$, current closure)",
     [("98%", OK), ("91%", OK), ("66%", NO), ("$-41$%", NO), ("–", NA), ("5.4 (closure)", OK), ("–", NA), ("under", NO)]),
    ("Measured $B$",
     [("48%", NO), ("94%", OK), ("52%", NO), ("$-0.3$%", OK), ("$-0.9$%", OK), ("1.3–4.2", NO), ("$4\\times10^4$", NO), ("under", NO)]),
    ("Enlarged potential",
     [("46%", NO), ("82%", OK), ("97%", OK), ("$-1.1$%", OK), ("$-1.8$%", OK), ("$-2.3$–3.3", NO), ("$4\\times10^4$", NO), ("under", NO)]),
    ("No density data",
     [("31%", NO), ("86%", OK), ("97%", OK), ("$-1.3$%", OK), ("$+3.9$%", OK), ("$-1.8$–3.3", NO), ("$2\\times10^4$", NO), ("20%", MID)]),
    ("A: uncertainty weights + $p_e$",
     [("38%", NO), ("90%", OK), ("96%", OK), ("$-0.7$%", OK), ("$+3.6$%", OK), ("$-1.5$–2.9", NO), ("$2\\times10^4$", NO), ("5%", OK)]),
    ("B: + local field-aligned residual",
     [("25%", NO), ("87%", OK), ("95%", OK), ("$-2.2$%", OK), ("$+0.7$%", OK), ("$-0.3$–2.5", NO), ("$3\\times10^3$", NO), ("16%", OK)]),
    ("C: + current (5.4 A)",
     [("10%", NO), ("88%", OK), ("97%", OK), ("$-2.7$%", OK), ("$-2.3$%", OK), ("5.1–5.4", OK), ("$2\\times10^3$", NO), ("66%", NO)]),
    ("Electron-only, local",
     [("10%", NO), ("88%", OK), ("97%", OK), ("$-3.4$%", OK), ("$-2.8$%", OK), ("5.3–5.4", OK), ("$2\\times10^3$", NO), ("frozen", NA)]),
    ("Electron-only, mobility-weighted",
     [("10%", NO), ("88%", OK), ("81%", NO), ("$-6.0$%", NO), ("$-4.6$%", OK), ("4.6–5.3", NO), ("113", NO), ("frozen", NA)]),
    ("Electron-only, tensor + FLI",
     [("10%", NO), ("84%", OK), ("97%", OK), ("$-5.9$%", NO), ("$-4.5$%", OK), ("5.1–5.5", OK), ("$1.3\\times10^3$", NO), ("frozen", NA)]),
    ("Coupled, tensor + FLI",
     [("6%", NO), ("91%", OK), ("97%", OK), ("$-3.6$%", OK), ("$-1.5$%", OK), ("5.3–5.9", MID), ("$1.4\\times10^3$", NO), ("83%", NO)]),
]
groups = [("Data agreement", 3), ("Conservation", 2), ("Current", 1), ("Electron mom.", 1), ("Ion energy", 1)]
heads = ["$n_i$", "$T_e$", "$V_p$", "Ions", "Neutrals", "$I_d$ (A)", "$I_{DD}$ (A)", "$W_{\\rm res}/K$"]
crit = ["$\\geq$80%", "$\\geq$80%", "$\\geq$90%", "$\\pm$5%", "$\\pm$5%", "5.4 A $\\pm$10%", "$\\leq 2I_d$", "$\\leq$20%"]

wNo, wName = 0.30, 2.35
wc = [0.62, 0.62, 0.62, 0.70, 0.70, 0.95, 0.85, 0.72]
rh, hh = 0.30, 0.30
W = wNo + wName + sum(wc)
nr = len(rows)
Htab = 3 * hh + nr * rh
Hbot = 1.70
fig = plt.figure(figsize=(W + 0.1, Htab + Hbot + 0.1))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.05, W + 0.05)
ax.set_ylim(-Hbot - 0.05, Htab + 0.05)
ax.axis("off")
xs = [0, wNo, wNo + wName]
for w in wc:
    xs.append(xs[-1] + w)
top = Htab
LW = 0.6

def box(x0, y0, w, h, fc="white", lw=LW):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fc, edgecolor="#333333", linewidth=lw))

# header
hdr = "#eef1f5"
box(0, top - 3 * hh, wNo, 3 * hh, hdr); ax.text(wNo / 2, top - 1.5 * hh, "No.", ha="center", va="center", weight="bold")
box(wNo, top - 3 * hh, wName, 3 * hh, hdr); ax.text(wNo + 0.08, top - 1.5 * hh, "Formulation", ha="left", va="center", weight="bold")
c = 0
for g, n in groups:
    x0, x1 = xs[2 + c], xs[2 + c + n]
    box(x0, top - hh, x1 - x0, hh, hdr)
    ax.text((x0 + x1) / 2, top - hh / 2, g, ha="center", va="center", weight="bold")
    c += n
for j in range(8):
    x0, x1 = xs[2 + j], xs[3 + j]
    box(x0, top - 2 * hh, x1 - x0, hh, hdr); ax.text((x0 + x1) / 2, top - 1.5 * hh, heads[j], ha="center", va="center")
    box(x0, top - 3 * hh, x1 - x0, hh, hdr); ax.text((x0 + x1) / 2, top - 2.5 * hh, crit[j], ha="center", va="center", fontsize=6.4)

y = top - 3 * hh
for i, (name, cells) in enumerate(rows):
    y0 = y - (i + 1) * rh
    box(0, y0, wNo, rh); ax.text(wNo / 2, y0 + rh / 2, str(i + 1), ha="center", va="center")
    box(wNo, y0, wName, rh); ax.text(wNo + 0.08, y0 + rh / 2, name, ha="left", va="center")
    for j, (txt, st) in enumerate(cells):
        x0, x1 = xs[2 + j], xs[3 + j]
        box(x0, y0, x1 - x0, rh, FILL[st])
        if st == NA:
            ax.text((x0 + x1) / 2, y0 + rh / 2, txt, ha="center", va="center", color="#555555")
        else:
            ax.text((x0 + x1) / 2 - 0.07, y0 + rh / 2, txt, ha="center", va="center")
            ax.text(x1 - 0.09, y0 + rh / 2, MARK[st], ha="center", va="center", fontsize=7.5, family="DejaVu Sans",
                    color={OK: "#2e7d32", MID: "#8a6d00", NO: "#b71c1c"}[st])
# separators: reference | data-constrained sequence | electron-only | coupled
for k in (1, 7, 10):
    yy = y - k * rh
    ax.plot([0, W], [yy, yy], color="#333333", lw=1.4)
box(0, y - nr * rh, W, Htab, "none", lw=1.0)

# bottom: legend + key findings
yb = -0.12
lx = 0.0
ax.text(lx, yb, "Legend", weight="bold", va="top")
leg = [(OK, "satisfies the criterion"), (MID, "within 1% of the threshold"), (NO, "does not satisfy the criterion"), (NA, "not evaluated or not applicable")]
for k, (st, t) in enumerate(leg):
    yy = yb - 0.26 - k * 0.24
    box(lx, yy - 0.09, 0.30, 0.18, FILL[st], lw=0.5)
    ax.text(lx + 0.15, yy, MARK[st] if st != NA else "–", ha="center", va="center", family="DejaVu Sans",
            color={OK: "#2e7d32", MID: "#8a6d00", NO: "#b71c1c", NA: "#555555"}[st])
    ax.text(lx + 0.40, yy, t, va="center")
ax.text(0.0, yb - 1.50, "$n_i,T_e,V_p$: fraction of points within the measurement bands. $I_{DD}$: largest momentum-implied current.",
        va="center", fontsize=6.4, color="#333333")

kx = 3.55
ax.text(kx, yb, "Key findings", weight="bold", va="top")
kf = [
    "No formulation satisfies all constraints.",
    "1. Measured ion density vs particle conservation: only the non-conserving",
    "    reference fits the density; conserving runs keep at most 48% of points.",
    "2. Imposed 5.4 A vs ion energy: runs carrying 5.4 A over-accelerate the ions;",
    "    energy-consistent runs carry at most 2.5–3.3 A.",
    "3. Field-aligned electron balance: $I_{DD}$ exceeds $2I_d$ in every formulation.",
]
for k, t in enumerate(kf):
    ax.text(kx, yb - 0.26 - k * 0.19, t, va="center", fontsize=6.8)

fig.savefig("compat_map.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("compat_map.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("ok")
