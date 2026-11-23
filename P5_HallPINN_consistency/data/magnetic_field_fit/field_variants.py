"""Roy 3 Nov: field-line-integral mismatch of the reconstructed data under physically admissible magnetic-field variants.
All variants are flux-function fits (div B = 0 identically) to the same 510 measured 1.6-kW points (x >= 10 mm)."""
import csv, json, copy
from pathlib import Path
import numpy as np, torch
import field as FM
HERE=Path(__file__).resolve().parent; OUT=HERE/'field_variants'; OUT.mkdir(exist_ok=True)
g=dict(r_inner_m=0.06,r_outer_m=0.085,channel_length_m=0.038)
rows=[q for q in csv.DictReader(open(HERE/'haas_vacuum_B.csv')) if float(q['power_kW'])==1.6]; assert len(rows)==510
Rs=[2.5,7.5,12.5,17.5,22.5]
def nearest(comp,R,x):
    c=[q for q in rows if q['component']==comp and float(q['R_mm_from_inner_wall'])==R]
    return min(c,key=lambda q:abs(float(q['x_mm_from_anode'])-x))
def pseudo(kind):
    P=[]
    for x in np.arange(0,9,1.0):
        for R in Rs:
            if kind=='radial': P.append(dict(component='axial',R_mm_from_inner_wall=str(R),x_mm_from_anode=str(x),B_G='0'))
            if kind=='const':
                for comp in ('axial','radial'):
                    q=nearest(comp,R,10.0); P.append(dict(component=comp,R_mm_from_inner_wall=str(R),x_mm_from_anode=str(x),B_G=q['B_G']))
    return P
models={'main_curl1':FM.fit(rows,g,1.0),'curl0p1':FM.fit(rows,g,0.1),
        'anode_radial':FM.fit(rows+pseudo('radial'),g,1.0),'anode_constant':FM.fit(rows+pseudo('const'),g,1.0)}
rng=np.random.default_rng(7)
for k in range(20):
    noisy=[dict(q,B_G=str(float(q['B_G'])+rng.normal(0,3.0))) for q in rows]
    models[f'noise3G_{k:02d}']=FM.fit(noisy,g,1.0)
# ---- data fields (same as boltz_data.py) ----
dr=list(csv.DictReader(open(HERE/'code/haas_gallimore_p5_5p4A.csv')))
X=sorted({float(r['x_mm']) for r in dr if float(r['x_mm'])<=38.0}); R=sorted({float(r['r_mm']) for r in dr})
G={q:np.full((len(X),len(R)),np.nan) for q in ('ni','Te','Vp')}
for r in dr:
    if float(r['x_mm'])<=38: G[r['quantity']][X.index(float(r['x_mm'])),R.index(float(r['r_mm']))]=float(r['value'])
X=np.array(X);R=np.array(R)
def interp(q,x,r):
    x=np.clip(x,X[0],X[-1]);r=np.clip(r,R[0],R[-1])
    i=np.clip(np.searchsorted(X,x)-1,0,len(X)-2);j=np.clip(np.searchsorted(R,r)-1,0,len(R)-2)
    tx=(x-X[i])/(X[i+1]-X[i]);tr=(r-R[j])/(R[j+1]-R[j]);gg=G[q]
    return (1-tx)*(1-tr)*gg[i,j]+tx*(1-tr)*gg[i+1,j]+(1-tx)*tr*gg[i,j+1]+tx*tr*gg[i+1,j+1]
def mismatch(model):
    seeds=np.arange(12,37,2.0); ns=len(seeds); ds=0.05
    def bdir(x,r):
        m=FM.evaluate(x/1000,0.06+r/1000,model); n=np.hypot(m['Bx'],m['Br']); return m['Bx']/n,m['Br']/n
    paths={}
    for sgn in (-1,1):
        x=seeds.copy(); r=np.full(ns,12.5); act=np.ones(ns,bool); P=[[(a,b)] for a,b in zip(x,r)]
        for _ in range(4000):
            if not act.any(): break
            f=lambda x,r: tuple(sgn*v for v in bdir(x,r))
            k1=f(x,r);k2=f(x+ds/2*k1[0],r+ds/2*k1[1]);k3=f(x+ds/2*k2[0],r+ds/2*k2[1]);k4=f(x+ds*k3[0],r+ds*k3[1])
            xn=x+ds/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);rn=r+ds/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            inside=(X[0]<=xn)&(xn<=X[-1])&(R[0]<=rn)&(rn<=R[-1])
            act=act&inside
            for k in np.where(act)[0]: P[k].append((xn[k],rn[k]))
            x=np.where(act,xn,x);r=np.where(act,rn,r)
        paths[sgn]=P
    out=[]
    for k in range(ns):
        Q=np.array(paths[-1][k][::-1]+paths[1][k][1:])
        n=interp('ni',Q[:,0],Q[:,1]);T=interp('Te',Q[:,0],Q[:,1]);V=interp('Vp',Q[:,0],Q[:,1])
        out.append(float((V[-1]-V[0])-np.sum(np.diff(n*T)/(0.5*(n[1:]+n[:-1])))))
    return out
seeds=list(np.arange(12,37,2.0)); res={}
for k,mdl in models.items():
    y=np.array([float(q['B_G'])*1e-4 for q in rows]); x=np.array([float(q['x_mm_from_anode'])/1000 for q in rows]); r=0.06+np.array([float(q['R_mm_from_inner_wall'])/1000 for q in rows])
    ev=FM.evaluate(x,r,mdl); fitv=np.where(np.array([q['component']=='axial' for q in rows]),ev['Bx'],ev['Br'])
    near={}
    for xa in (2.0,5.0,8.0):
        e=FM.evaluate(np.array([xa/1000]),np.array([0.0725]),mdl); near[f'x{xa:g}mm']=dict(B_G=float(np.hypot(e['Bx'][0],e['Br'][0])*1e4),b_x=float(e['Bx'][0]/np.hypot(e['Bx'][0],e['Br'][0])))
    mm=mismatch(mdl)
    res[k]=dict(data_rms_G=float(np.sqrt(np.mean((fitv-y)**2))*1e4),curl_loss=mdl['curl_loss'],near_anode_mid_radius=near,mismatch_V=mm,
                upstream_12_18mm=[round(v,1) for v in mm[:4]],downstream_20_36mm_max_abs=float(max(abs(v) for v in mm[4:])))
    if not k.startswith('noise'): json.dump(mdl,open(OUT/f'field_{k}.json','w'))
noise=np.array([res[k]['mismatch_V'] for k in res if k.startswith('noise')])
summary=dict(variants={k:v for k,v in res.items() if not k.startswith('noise')},
             noise3G_ensemble=dict(n=len(noise),seed_x_mm=seeds,mean_V=noise.mean(0).round(1).tolist(),std_V=noise.std(0).round(1).tolist(),
                                   min_V=noise.min(0).round(1).tolist(),max_V=noise.max(0).round(1).tolist()))
json.dump(summary,open(HERE/'field_variants.json','w'),indent=1)
for k,v in summary['variants'].items(): print(k,'fitRMS %.2f G'%v['data_rms_G'],'curl %.2e'%v['curl_loss'],'near',{a:(round(b['B_G']),round(b['b_x'],2)) for a,b in v['near_anode_mid_radius'].items()},'mismatch',[round(z,1) for z in v['mismatch_V']])
e=summary['noise3G_ensemble']; print('noise ensemble mean',e['mean_V']); print('noise ensemble std',e['std_V'])
