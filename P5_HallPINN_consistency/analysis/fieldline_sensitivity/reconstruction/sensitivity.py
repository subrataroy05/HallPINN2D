"""Sensitivity of the field-line mismatch M = dVp - int dp_e/(e n_e) of the reconstructed measurements to
(1) the radial interpolation between the five probe radii and (2) the probe-position uncertainty.
Same field lines, data and quadrature as independent_checks.py (baseline reproduced exactly by 'linear')."""
import csv, json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import field as FM

H = Path(__file__).resolve().parent
model = json.load(open(H / 'field_main.json'))
rows = list(csv.DictReader(open(H / 'haas_gallimore_p5_5p4A.csv')))
XF = np.array(sorted({float(r['x_mm']) for r in rows}))          # full probe range (to 80 mm), used for shifts
R = np.array(sorted({float(r['r_mm']) for r in rows}))
GF = {q: np.zeros((len(XF), len(R))) for q in ('ni', 'Te', 'Vp')}
for r in rows:
    GF[r['quantity']][list(XF).index(float(r['x_mm'])), list(R).index(float(r['r_mm']))] = float(r['value'])
X = XF[XF <= 38]

def bdir(x, r):
    m = FM.evaluate(x / 1000, 0.06 + r / 1000, model); n = np.hypot(m['Bx'], m['Br']); return m['Bx'] / n, m['Br'] / n

def trace(seeds, ds=0.05):
    out = {}
    for sgn in (-1, 1):
        x = seeds.copy(); r = np.full(len(seeds), 12.5); act = np.ones(len(x), bool); P = [[(a, b)] for a, b in zip(x, r)]
        for _ in range(4000):
            if not act.any(): break
            f = lambda x, r: tuple(sgn * v for v in bdir(x, r))
            k1 = f(x, r); k2 = f(x + ds / 2 * k1[0], r + ds / 2 * k1[1]); k3 = f(x + ds / 2 * k2[0], r + ds / 2 * k2[1]); k4 = f(x + ds * k3[0], r + ds * k3[1])
            xn = x + ds / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); rn = r + ds / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            act = act & (X[0] <= xn) & (xn <= X[-1]) & (R[0] <= rn) & (rn <= R[-1])
            for k in np.where(act)[0]: P[k].append((xn[k], rn[k]))
            x = np.where(act, xn, x); r = np.where(act, rn, r)
        out[sgn] = P
    return [np.array(out[-1][k][::-1] + out[1][k][1:]) for k in range(len(seeds))]

seeds = np.arange(12, 37, 2.0); lines = trace(seeds)
XS = {id(P): P[:, 0].copy() for P in lines}; RS = {id(P): P[:, 1].copy() for P in lines}

def columns_at(G, x, dx=None):
    """Values of each radius column at axial positions x (linear in x); dx: per-radius axial shift of the probe (mm)."""
    out = np.empty((len(x), len(R)))
    if dx is None:
        key=(id(x),id(G))
        if key in _CC: return _CC[key]
    for j in range(len(R)):
        s = 0.0 if dx is None else dx[j]
        out[:, j] = np.interp(x - s, XF, G[:, j])   # measured at nominal x_true = x_reported + s
    if dx is None: _CC[(id(x),id(G))] = out
    return out
_CC = {}

I5 = np.eye(len(R))
def rweights(r, method):
    """Weights (npts, 5) of the linear radial interpolation operators."""
    if method == 'linear':
        return np.array([np.interp(r, R, I5[k]) for k in range(len(R))]).T
    if method == 'spline':
        return CubicSpline(R, I5, axis=0, bc_type='natural')(r)
    if method == 'quadratic':
        V = np.vander(R, 3); A = np.linalg.pinv(V)          # least-squares quadratic through 5 points
        return np.vander(r, 3) @ A
    if method == 'nearest':
        return I5[np.argmin(abs(R[None, :] - r[:, None]), 1)]
    raise ValueError(method)
_WC = {}
def radial(cols, r, method, logq=False):
    v = np.log(cols) if logq else cols
    if method == 'pchip':                                    # nonlinear: point by point
        res = np.array([PchipInterpolator(R, vv)(rr) for rr, vv in zip(r, v)])
    else:
        key = (id(r), method)
        if key not in _WC: _WC[key] = rweights(r, method)
        res = np.sum(v * _WC[key], 1)
    return np.exp(res) if logq else res

def M_line(P, method='linear', logn=False, dx=None, G=GF):
    x, r = XS[id(P)], RS[id(P)]
    n = radial(columns_at(G['ni'], x, dx), r, method, logq=logn)
    T = radial(columns_at(G['Te'], x, dx), r, method)
    V = radial(columns_at(G['Vp'], x, dx), r, method)
    p = n * T
    return (V[-1] - V[0]) - np.sum(np.diff(p) / (0.5 * (n[1:] + n[:-1])))

def M_all(**kw): return np.array([M_line(P, **kw) for P in lines])

out = {'seeds_mm': seeds.tolist()}
# (1) radial interpolation
interp = {}
for method in ('linear', 'pchip', 'spline', 'quadratic', 'nearest'):
    interp[method] = M_all(method=method).round(2).tolist()
interp['linear_log_density'] = M_all(method='linear', logn=True).round(2).tolist()
interp['pchip_log_density'] = M_all(method='pchip', logn=True).round(2).tolist()
out['radial_interpolation'] = interp

# (2) probe-position uncertainty: axial position +-1.5 mm (Haas 2001), per probe radius
rng = np.random.default_rng(7); N = 2000
pos = {}
for label, draw in (('gauss_2sig', lambda: rng.normal(0, 0.75, len(R))),      # +-1.5 mm read as 2 sigma
                    ('uniform_1p5', lambda: rng.uniform(-1.5, 1.5, len(R))),  # worst case, uniform within +-1.5 mm
                    ('common_shift', lambda: np.full(len(R), rng.uniform(-1.5, 1.5)))):
    S = np.array([M_all(dx=draw()) for _ in range(N)])
    pos[label] = dict(mean=S.mean(0).round(2).tolist(), std=S.std(0).round(2).tolist(),
                      p05=np.percentile(S, 5, 0).round(2).tolist(), p95=np.percentile(S, 95, 0).round(2).tolist(),
                      min=S.min(0).round(2).tolist())
# deterministic extremes: alternating shifts of +-1.5 mm across radii
for sgn in (1, -1):
    pos[f'alternating_{"+" if sgn>0 else "-"}'] = M_all(dx=sgn * 1.5 * np.array([1, -1, 1, -1, 1])).round(2).tolist()
    pos[f'tilt_{"+" if sgn>0 else "-"}'] = M_all(dx=sgn * 1.5 * np.linspace(-1, 1, len(R))).round(2).tolist()
out['probe_position'] = pos

# (3) combined: interpolation method drawn at random + per-radius Gaussian position error
meths = ['linear', 'spline', 'quadratic', 'linear']
S = np.array([M_all(method=meths[rng.integers(4)], logn=bool(rng.integers(2)), dx=rng.normal(0, 0.75, len(R))) for _ in range(N)])
out['combined'] = dict(mean=S.mean(0).round(2).tolist(), std=S.std(0).round(2).tolist(),
                       p05=np.percentile(S, 5, 0).round(2).tolist(), p95=np.percentile(S, 95, 0).round(2).tolist())
json.dump(out, open(H / 'sensitivity.json', 'w'), indent=1)
print('seeds', seeds.tolist())
for k, v in interp.items(): print(f'{k:20s}', v)
for k, v in pos.items(): print(k, v if isinstance(v, list) else {kk: vv[:6] for kk, vv in v.items()})
print('combined', {kk: vv[:6] for kk, vv in out['combined'].items()})
