"""Read-only post-analysis of bfield_2d final checkpoint (no training).
1) parallel/perpendicular split of I_DD on the FINAL bfield_2d fields.
2) ion x-momentum term decomposition along x at 5 radii; where the residual sits.
3) Vp model vs data at mid-radius alongside the ion-momentum residual."""
import sys, json, csv
from pathlib import Path
H=Path(__file__).resolve().parent; B2=H.parent/'bfield_2d'
sys.path.insert(0,str(B2))
import numpy as np, torch
import diagnostics as D
a=D.a; U=a.utils
torch.set_num_threads(4)
m,d,p,_=a.load_model(B2/'output/checkpoint.pt')
out={}
# 1) split
rows=D.sections(m,d,p,np.linspace(0,1,40),force_2d=True)
with open(H/'split_final.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out['split']=[{k:round(v,4) for k,v in r.items()} for r in rows if round(r['x_hat']*39) in (0,4,8,13,20,29,35,39)]
xg,rg=a.grid(d); q=D.local(m,p,xg,rg,True); F=q['Fpar_V_m']
out['Fpar_RMS_V_m']=float(np.sqrt(np.mean(F**2)))
# 2) ion momentum terms
def g(y,z): return torch.autograd.grad(y.sum(),z,retain_graph=True,create_graph=False)[0]
def terms(x,r):
    x=x.clone().requires_grad_(True); r=r.clone().requires_grad_(True)
    phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=m(x,r)
    Si=U.compute_Si(nn_,ne,Te,p); coll=U.compute_collision_frequencies(ne,nn_,Te,p)
    tau=p.get('tau_ref',p['L_canal']/p['V_ref'])
    inertia=g(ni*Vix*Vix,x)+g(ni*Vix*Vir,r)+ni*Vix*Vir/r
    elec=ni*g(phi,x)
    edrag=-ni*(p['me']/p['mi'])*coll['nu_total']*tau*(Vex-Vix)
    nuin=U.compute_ion_neutral_frequency(nn_,Vix,p)*tau
    fric=ni*(p['mn']/p['mi'])*nuin*(Vix-p['Vn'])
    ioniz=-Si*p['Vn']
    res=inertia+elec+edrag+fric+ioniz
    T=dict(inertia=inertia,electric=elec,e_drag=edrag,in_friction=fric,ionization=ioniz,residual=res,phi_V=phi*p['phi_ref'],Vix_kms=Vix*p['V_ref']/1000,ni=ni)
    return {k:v.detach().numpy() for k,v in T.items()}
xs=np.linspace(0,1,201); prof=[]
for R in [2.5,7.5,12.5,17.5,22.5]:
    rr=d['r_min']+R/1000/p['L_canal']
    t=terms(torch.tensor(xs),torch.full((201,),rr))
    for i,x in enumerate(xs): prof.append(dict(R_mm=R,x_hat=x,**{k:float(v[i]) for k,v in t.items()}))
with open(H/'ion_momentum_terms.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(prof[0]));w.writeheader();w.writerows(prof)
t=terms(xg,rg); res=t['residual']; X=xg.numpy()
bins=[(0,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.01)]
out['ion_mom_res_share_by_x']={f'{lo}-{hi}':float((res[(X>=lo)&(X<hi)]**2).sum()/(res**2).sum()) for lo,hi in bins}
out['ion_mom_res_RMS']=float(np.sqrt((res**2).mean()))
mid=[v for v in prof if v['R_mm']==12.5]
out['midradius_terms']=[{k:round(v[k],4) for k in ['x_hat','inertia','electric','e_drag','in_friction','ionization','residual','phi_V','Vix_kms']} for v in mid if round(v['x_hat']*200)%20==0]
json.dump(out,open(H/'post.json','w'),indent=1); print(json.dumps(out,indent=1))
