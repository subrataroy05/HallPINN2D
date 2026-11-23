"""Extract the measured axial profiles (Vp, ni, Te, Br) at five radii, 300 V / 5.4 A, from the vector figures
5-9 of Haas & Gallimore AIAA-2000-3422. Curves are polylines whose vertices are the data points; axes are
calibrated from the tick-label positions of each figure."""
import sys, json, csv
import numpy as np
import pymupdf

pdf = sys.argv[1]; outcsv = sys.argv[2]
doc = pymupdf.open(pdf)
FIGS = [(5, 0, 2.5), (5, 1, 7.5), (6, 0, 12.5), (6, 1, 17.5), (7, 0, 22.5)]   # (page index, top/bottom, radius mm)

def spans(pg):
    out = []
    for b in pg.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                t = s['text'].strip()
                if t: out.append((t, pymupdf.Rect(s['bbox'])))
    return out

def fit(pairs):
    a = np.array(pairs); A = np.vstack([a[:, 0], np.ones(len(a))]).T
    k, c = np.linalg.lstsq(A, a[:, 1], rcond=None)[0]
    resid = a[:, 1] - (k * a[:, 0] + c)
    return k, c, float(np.abs(resid).max())

def num(t):
    t = t.replace('x10', '')
    try: return float(t)
    except ValueError: return None

def chains(pg, clip):
    """Long polylines (data curves) and their marker type, from all straight segments inside clip."""
    segs, short, circ, rects = [], [], [], []
    for d in pg.get_drawings():
        for it in d['items']:
            if it[0] == 'c':
                circ.append(d['rect']); break
            if it[0] == 're':
                rects.append(it[1]); continue
            if it[0] != 'l': continue
            p, q = it[1], it[2]
            if not (clip.contains(p) and clip.contains(q)): continue
            L = abs(q - p)
            if L < 4.2 and L > 0.3: short.append(((p.x, p.y), (q.x, q.y)))
            if 0.8 < L < 60 and abs(q.x - p.x) > 0.5: segs.append(((p.x, p.y), (q.x, q.y)))
    segs = [(a, b) if a[0] <= b[0] else (b, a) for a, b in segs]
    used = [False] * len(segs)
    A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs])
    out = []
    heads = [i for i in range(len(segs)) if np.min(np.hypot(*(B - A[i]).T)) > 0.15]
    for h in heads:
        if used[h]: continue
        pts = [segs[h][0], segs[h][1]]; used[h] = True; cur = np.array(segs[h][1])
        while True:
            dd = np.hypot(*(A - cur).T); dd[np.array(used)] = 1e9
            k = int(np.argmin(dd))
            if dd[k] > 0.15: break
            used[k] = True; pts.append(segs[k][1]); cur = np.array(segs[k][1])
        if len(pts) >= 5: out.append(np.array(pts))
    def marker(ch):
        votes = {'circle': 0, 'square': 0, 'triangle': 0, 'diamond': 0}
        for v in ch:
            if any(abs((r.x0 + r.x1) / 2 - v[0]) < 1.2 and abs((r.y0 + r.y1) / 2 - v[1]) < 1.2 and r.width < 4 for r in circ):
                votes['circle'] += 1; continue
            if any(abs((r.x0 + r.x1) / 2 - v[0]) < 2.5 and abs((r.y0 + r.y1) / 2 - v[1]) < 2.5 for r in rects):
                votes['square'] += 1; continue
            near = [s for s in short if max(np.hypot(s[0][0] - v[0], s[0][1] - v[1]), np.hypot(s[1][0] - v[0], s[1][1] - v[1])) < 2.6]
            diag = [s for s in near if abs(abs(s[1][0] - s[0][0]) - abs(s[1][1] - s[0][1])) < 0.4 * abs(s[1][0] - s[0][0]) + 0.05]
            horiz = [s for s in near if abs(s[1][1] - s[0][1]) < 0.15 and abs(s[1][0] - s[0][0]) > 1.5]
            vert = [s for s in near if abs(s[1][0] - s[0][0]) < 0.15 and abs(s[1][1] - s[0][1]) > 1.5]
            if len(horiz) >= 2 and len(vert) >= 2: votes['square'] += 1
            elif len(diag) >= 4: votes['diamond'] += 1
            else:
                slant = [s for s in near if abs(s[1][1] - s[0][1]) > 0.5 and abs(s[1][0] - s[0][0]) > 0.3]
                if len(horiz) == 1 and len(slant) >= 2: votes['triangle'] += 1
        best = max(votes, key=votes.get)
        return (best if votes[best] > 0 else None), votes
    return [(c, *marker(c)) for c in out]

ref = {}
for r in csv.DictReader(open(sys.argv[3])):   # current reconstruction, used only to label curves
    ref.setdefault((float(r['r_mm']), r['quantity']), []).append((float(r['x_mm']), float(r['value'])))

rows = []; report = {}
for pi, half, Rmm in FIGS:
    pg = doc[pi]; S = spans(pg)
    ymid = pg.rect.height / 2
    sel = [(t, b) for t, b in S if (b.y0 < ymid) == (half == 0)]
    # x axis: labels 0..80 on the lowest row of numbers
    xl = [(b.x0 + b.x1) / 2 for t, b in sel if num(t) is not None]
    rowy = max(b.y0 for t, b in sel if t in ('0', '80') and b.x0 > 140 and b.x1 < 470)
    xs = [((b.x0 + b.x1) / 2, num(t)) for t, b in sel if abs(b.y0 - rowy) < 1 and num(t) is not None and 140 < b.x0 < 470]
    kx, cx, ex = fit(xs)
    def col(xmin, xmax):
        return [((b.y0 + b.y1) / 2, num(t)) for t, b in sel if xmin <= b.x0 <= xmax and num(t) is not None and b.y0 < rowy - 2]
    axes = {'Vp': col(100, 125), 'Br': col(135, 150), 'ni': col(460, 480), 'Te': col(490, 510)}
    cal = {q: fit(v) for q, v in axes.items()}
    clip = pymupdf.Rect(150, min(b.y0 for t, b in sel if t == '180') - 5, 470, rowy - 6)
    C = chains(pg, clip)
    MAP = {'circle': 'Br', 'square': 'Vp', 'triangle': 'ni', 'diamond': 'Te'}
    fig = {}
    for ch, mk, votes in C:
        if mk is None: continue
        q = MAP[mk]
        k, c, e = cal[q]
        x = kx * ch[:, 0] + cx; y = k * ch[:, 1] + c
        if q == 'ni': y = y * 1e18
        fig.setdefault(q, []).append((x, y, votes))
    for q in list(fig):
        X_ = np.concatenate([f[0] for f in fig[q]]); Y_ = np.concatenate([f[1] for f in fig[q]])
        o = np.argsort(X_); X_, Y_ = X_[o], Y_[o]
        keep = np.r_[True, np.diff(X_) > 0.3]
        fig[q] = (X_[keep], Y_[keep], [f[2] for f in fig[q]])
    report[Rmm] = {q: (len(v[0]), round(float(v[0].min()), 1), round(float(v[0].max()), 1), v[2]) for q, v in fig.items()}
    report[Rmm]['axis_fit_max_resid_pt'] = {q: round(cal[q][2], 2) for q in cal} | {'x': round(ex, 2)}
    for q in ('Vp', 'ni', 'Te'):
        if q not in fig: continue
        x, y, _ = fig[q]
        for xx, yy in zip(x, y): rows.append(dict(r_mm=Rmm, quantity=q, x_mm=round(float(xx), 3), value=float(yy)))

with open(outcsv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['x_mm', 'r_mm', 'quantity', 'value']); w.writeheader(); w.writerows(rows)
print(json.dumps(report, indent=1, default=str))
