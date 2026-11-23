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
out={}
for name,xv in [('anode',0.05),('B_peak',xB),('exit',0.97)]:
  rows=[]
  for frac in (0.1,0.5,0.9):
    x=torch.tensor([xv],dtype=torch.float64).requires_grad_(True); r=torch.tensor([d['r_min']+frac*(d['r_max']-d['r_min'])]).requires_grad_(True)
    phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=w(x,r)
    Bx,Br=Bfield(x,r); Bm=torch.hypot(Bx,Br); bx_,br_=Bx/Bm,Br/Bm
    coll=U.compute_collision_frequencies(ne,nn_,Te,p); om=p['e_over_me']*Bm
    nu_cl=coll['nu_total']; nu_B=p['alpha_B']*om
    inx=mu*(ne*Vex*g(Vex,x)+ne*Ver*g(Vex,r)); inr=mu*(ne*Vex*g(Ver,x)+ne*Ver*g(Ver,r))
    inertia=bx_*inx+br_*inr
    electric=-ne*(bx_*g(phi,x)+br_*g(phi,r))
    pe=ne*Te; pressure=bx_*g(pe,x)+br_*g(pe,r)
    Vpar=bx_*Vex+br_*Ver
    drag_cl=ne*mu*nu_cl*tau*Vpar; drag_B=ne*mu*nu_B*tau*Vpar
    tot=inertia+electric+pressure+drag_cl+drag_B
    f=lambda t: float(t.detach())*S
    neSI=float(ne.detach())*p['n_ref']
    # drift-diffusion parallel velocity implied by (electric+pressure) with full nu and classical nu only
    force_ep=float((electric+pressure).detach())*S
    V_DD_full=-force_ep/(neSI*p['me']*float((nu_cl+nu_B).detach())); V_DD_cl=-force_ep/(neSI*p['me']*float(nu_cl.detach()))
    rows.append(dict(r_frac=frac,B_G=float(Bm.detach())*1e4,b_x=float(bx_.detach()),ne_m3=neSI,Te_eV=float(Te.detach())*p['phi_ref'],
      nu_ei=float(coll['nu_ei'].detach()),nu_en=float(coll['nu_en'].detach()),nu_Bohm=float(nu_B.detach()),
      inertia_N_m3=f(inertia),electric_N_m3=f(electric),pressure_N_m3=f(pressure),drag_classical_N_m3=f(drag_cl),drag_Bohm_N_m3=f(drag_B),residual_N_m3=f(tot),
      residual_V_m=f(tot)/(p['e_phys']*neSI),electric_V_m=f(electric)/(p['e_phys']*neSI),pressure_V_m=f(pressure)/(p['e_phys']*neSI),
      Vpar_network_m_s=float(Vpar.detach())*p['V_ref'],Vpar_DD_fullnu_m_s=V_DD_full,Vpar_DD_classical_m_s=V_DD_cl))
  out[name]=dict(x_hat=xv,rows=rows)
json.dump(out,open(HERE/'eterms_frozenC.json','w'),indent=1)
for k,v in out.items():
  print('==',k,'x_hat=%.3f'%v['x_hat'])
  for r in v['rows']: print({a_:('%.3g'%b_) for a_,b_ in r.items()})
