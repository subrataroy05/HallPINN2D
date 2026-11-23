"""Roy 2 Nov (22:03): dimensional field-aligned electron momentum terms at three locations
on the frozenC_electron final checkpoint, with the parallel drag split into classical and Bohm parts."""
import sys, copy, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; F=HERE.parent/'frozenC_electron'; sys.path.insert(0,str(F)); sys.path.insert(0,str(F/'code'))
import numpy as np, torch
import diagnostics as D, experiment as E
from field import B as Bfield
a=D.a; U=a.utils; torch.set_num_threads(2)
m,d0,p0,_=a.load_model(E.BASE)
c=torch.load(F/'output/checkpoint.pt',map_location='cpu',weights_only=False); d,p=c['domain'],c['physics']
m.phi_left=d['phi_left']; fr=copy.deepcopy(m); fr.eval()
m.load_state_dict({k[10:]:v for k,v in c['model_state_dict'].items() if k.startswith('train_net.')},strict=True)
w=E.FrozenHeavy(m,fr); w.eval()
g=lambda y,z: torch.autograd.grad(y.sum(),z,retain_graph=True)[0]
S=p['n_ref']*p['e_phys']*p['phi_ref']/p['L_canal']   # N/m^3 per normalized force unit
mu=p['me']/p['mi']; tau=p['tau_ref']
rmid=0.5*(d['r_min']+d['r_max'])
# B peak along mid-radius
xs=torch.linspace(0,1,1001); bx,br=Bfield(xs,torch.full_like(xs,rmid)); xB=float(xs[torch.hypot(bx,br).argmax()])

import math
res=[]
for frac in (0.1,0.5,0.9):
  for xv in np.linspace(0.02,0.98,49):
    x=torch.tensor([xv],dtype=torch.float64).requires_grad_(True); r=torch.tensor([d['r_min']+frac*(d['r_max']-d['r_min'])]).requires_grad_(True)
    phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=w(x,r)
    Bx,Br=Bfield(x,r); Bm=torch.hypot(Bx,Br); bx_,br_=Bx/Bm,Br/Bm
    neSI=float(ne.detach())*p['n_ref']
    inx=mu*(ne*Vex*g(Vex,x)+ne*Ver*g(Vex,r)); inr=mu*(ne*Vex*g(Ver,x)+ne*Ver*g(Ver,r))
    cent=-mu*ne*Vet**2/r   # radial centrifugal (normalized), acts as -m n Vth^2/r on the LHS
    electric=-ne*(bx_*g(phi,x)+br_*g(phi,r)); pe=ne*Te; pressure=bx_*g(pe,x)+br_*g(pe,r)
    toV=lambda t: float(t.detach())*S/(p['e_phys']*neSI)
    res.append(dict(r_frac=frac,x_hat=float(xv),b_r=float(br_.detach()),Vet_m_s=float(Vet.detach())*p['V_ref'] if 'V_ref' in p else float(Vet.detach()),
      inertia_old_V_m=toV(bx_*inx+br_*inr),centrifugal_par_V_m=toV(br_*cent),electric_V_m=toV(electric),pressure_V_m=toV(pressure),
      residual_old_V_m=None))
json.dump(res,open(HERE/'centrifugal_frozenC.json','w'),indent=1)
print('done')
