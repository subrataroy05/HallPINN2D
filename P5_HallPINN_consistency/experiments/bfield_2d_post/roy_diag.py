"""Roy 1 Nov requests, read-only, on a given experiment folder's final checkpoint.
(1) field-aligned force imbalance R_par = b.grad(phi) - b.grad(neTe)/ne  [V/m]
(2) L_par = <((Ve_par - Ve_par^DD)/V_ref)^2>
(3) axial ion momentum budget at 40 stations (cross-section integrated) vs Vp error
(+) potential-envelope saturation |tanh(raw0)|"""
import sys, json, csv
from pathlib import Path
F=Path(__file__).resolve().parent.parent/sys.argv[1]; sys.path.insert(0,str(F))
import numpy as np, torch
import diagnostics as D
a=D.a; U=a.utils; torch.set_num_threads(2)
m,d,p,_=a.load_model(F/'output/checkpoint.pt')
O=Path(__file__).resolve().parent/('roy_'+sys.argv[1]); O.mkdir(exist_ok=True)
nr=48; zg,wg=np.polynomial.legendre.leggauss(nr)
rad=(d['r_min']+d['r_max'])/2+(d['r_max']-d['r_min'])*zg/2; wr=wg*(d['r_max']-d['r_min'])/2*2*np.pi*rad
def gr(y,z): return torch.autograd.grad(y.sum(),z,retain_graph=True)[0]
rows=[]; Lpar_num=0; Lpar_n=0
for xs in np.linspace(0,1,40):
    x=torch.full((nr,),float(xs)); r=torch.tensor(rad)
    q=D.local(m,p,x,r,True)                       # 3x3 DD solve (with pressure), b components
    xx=x.clone().requires_grad_(True); rr=r.clone().requires_grad_(True)
    phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=m(xx,rr)
    Si=U.compute_Si(nn_,ne,Te,p); coll=U.compute_collision_frequencies(ne,nn_,Te,p)
    tau=p.get('tau_ref',p['L_canal']/p['V_ref'])
    T=dict(convective=gr(ni*Vix*Vix,xx)+gr(ni*Vix*Vir,rr)+ni*Vix*Vir/rr,
           electric=ni*gr(phi,xx),
           e_drag=-ni*(p['me']/p['mi'])*coll['nu_total']*tau*(Vex-Vix),
           in_exchange=ni*(p['mn']/p['mi'])*U.compute_ion_neutral_frequency(nn_,Vix,p)*tau*(Vix-p['Vn']),
           ionization=-Si*p['Vn'])
    T={k:v.detach().numpy() for k,v in T.items()}; res=sum(T.values())
    Vepar=(Vex.detach().numpy()*q['bx']+Ver.detach().numpy()*q['br'])
    Vddpar=q['vx']*q['bx']+q['vr']*q['br']
    lp=(Vepar-Vddpar)**2; Lpar_num+=float((lp*wr).sum()); Lpar_n+=float(wr.sum())
    row=dict(x_hat=float(xs),x_mm=float(xs*p['L_canal']*1000),
             Rpar_rms_V_m=float(np.sqrt((q['Fpar_V_m']**2*wr).sum()/wr.sum())),
             Rpar_mid_V_m=float(np.interp(0.5*(d['r_min']+d['r_max']),rad,q['Fpar_V_m'])),
             Lpar_station=float((lp*wr).sum()/wr.sum()))
    for k,v in T.items(): row['ion_'+k]=float((v*wr).sum())
    row['ion_residual']=float((res*wr).sum()); row['ion_residual_rms']=float(np.sqrt((res**2*wr).sum()/wr.sum()))
    rows.append(row)
# Vp error and envelope saturation at data points
data=U.load_haas_gallimore_data(str(F/'code/haas_gallimore_p5_5p4A.csv'),d,p,'cpu')
c=data['Vp']['coords']; obs=data['Vp']['values']
with torch.no_grad():
    pred=m(c[:,0],c[:,1])[0]; raw0=m.net(torch.stack([c[:,0],c[:,1]],1))[:,0]
err=((pred-obs)*p['phi_ref']).numpy(); xd=(c[:,0].numpy()); sat=np.abs(np.tanh((0.5 if "PHIFIX" in (F/"code/model.py").read_text() else 1.0)*raw0.numpy()))
vp=[dict(x_hat=float(x),Vp_err_mean_V=float(err[np.isclose(xd,x)].mean()),Vp_err_min_V=float(err[np.isclose(xd,x)].min()),tanh_raw0_max=float(sat[np.isclose(xd,x)].max())) for x in np.unique(np.round(xd,6))]
for name,R in [('stations.csv',rows),('vp_error.csv',vp)]:
    with open(O/name,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(R[0])); w.writeheader(); w.writerows(R)
summ=dict(Lpar=Lpar_num/Lpar_n, Rpar_rms_overall_V_m=float(np.sqrt(np.mean([r['Rpar_rms_V_m']**2 for r in rows]))),
          ion_residual_rms_overall=float(np.sqrt(np.mean([r['ion_residual_rms']**2 for r in rows]))),
          Vp_points_tanh_gt_0p95=int((sat>0.95).sum()), Vp_points=int(len(sat)))
json.dump(summ,open(O/'summary.json','w'),indent=1); print(json.dumps(summ,indent=1))
for r in rows[::4]+[rows[-1]]: print({k:round(v,4) for k,v in r.items()})
for v in vp[::3]: print({k:round(z,3) for k,z in v.items()})
