"""C5: axial ion energy budget along lines of constant r for a given run."""
import sys, json
from pathlib import Path
F=Path(__file__).resolve().parent.parent/sys.argv[1]; sys.path.insert(0,str(F))
import numpy as np, torch
import diagnostics as D
a=D.a; U=a.utils; torch.set_num_threads(2)
m,d,p,_=a.load_model(F/'output/checkpoint.pt')
gr=lambda y,z: torch.autograd.grad(y.sum(),z,retain_graph=True)[0]
T=p['phi_ref']; out={}
for f in (0.25,0.5,0.75):
    N=801; xs=np.linspace(0,1,N); rv=d['r_min']+f*(d['r_max']-d['r_min'])
    x=torch.tensor(xs).requires_grad_(True); r=torch.full((N,),rv).requires_grad_(True)
    phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=m(x,r)
    Si=U.compute_Si(nn_,ne,Te,p); coll=U.compute_collision_frequencies(ne,nn_,Te,p); tau=p.get('tau_ref',p['L_canal']/p['V_ref'])
    conv=gr(ni*Vix*Vix,x)+gr(ni*Vix*Vir,r)+ni*Vix*Vir/r
    el=ni*gr(phi,x); dr=-ni*(p['me']/p['mi'])*coll['nu_total']*tau*(Vex-Vix)
    ex=ni*(p['mn']/p['mi'])*U.compute_ion_neutral_frequency(nn_,Vix,p)*tau*(Vix-p['Vn']); io=-Si*p['Vn']
    R=(conv+el+dr+ex+io).detach().numpy(); n=ni.detach().numpy(); V=Vix.detach().numpy(); ph=phi.detach().numpy()
    tr=lambda y: float(np.trapz(y,xs))
    dKE=0.5*(V[-1]**2-V[0]**2)*T; Ework=-(ph[-1]-ph[0])*T; Rwork=tr(R/n)*T
    out[f'r_frac_{f}']=dict(phi_anode_V=float(ph[0]*T),phi_max_V=float(ph.max()*T),x_phimax=float(xs[ph.argmax()]),phi_exit_V=float(ph[-1]*T),
       Vix_exit_km_s=float(V[-1]*p['V_ref']/1e3),KE_exit_eV=float(0.5*V[-1]**2*T),max_available_drop_V=float((ph.max()-ph[-1])*T),
       dKE_eV=dKE,electric_work_eV=Ework,residual_work_eV=Rwork,other_physical_eV=dKE-Ework-Rwork,
       ion_res_rms_over_electric_rms=float(np.sqrt(np.mean(R**2))/np.sqrt(np.mean(el.detach().numpy()**2))))
print(json.dumps(out,indent=1)); json.dump(out,open(Path(__file__).parent/f'energy_{sys.argv[1]}.json','w'),indent=1)
