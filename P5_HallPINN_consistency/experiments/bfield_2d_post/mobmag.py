import sys, copy
from pathlib import Path
HERE=Path(__file__).resolve().parent; F=HERE.parent/'frozenC_electron'; sys.path.insert(0,str(F)); sys.path.insert(0,str(F/'code'))
import numpy as np, torch, diagnostics as D, experiment as E
from field import B as Bfield
a=D.a; U=a.utils; torch.set_num_threads(4)
m,_,_,_=a.load_model(E.BASE); c=torch.load(F/'output/checkpoint.pt',map_location='cpu',weights_only=False); d,p=c['domain'],c['physics']
m.phi_left=d['phi_left']; fr=copy.deepcopy(m); fr.eval()
m.load_state_dict({k[10:]:v for k,v in c['model_state_dict'].items() if k.startswith('train_net.')}); w=E.FrozenHeavy(m,fr)
x,r=a.grid(d); x=x.clone().requires_grad_(True); r=r.clone().requires_grad_(True)
eq=U.compute_equation_residuals(w,x,r,p)
phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=eq['fields']
Bx,By=Bfield(x,r); om=p['e_over_me']*torch.hypot(Bx,By)
nu_full=eq['coll']['nu_total']+p['alpha_B']*om; nu_cl=eq['coll']['nu_total']
RV=eq['elec_par'].detach()*p.get('E_ref_par',1200.)
for name,nu in (('full',nu_full),('classical',nu_cl)):
    u=(p['e_phys']*RV/(p['me']*nu.detach())).numpy()
    print(name,'RMS u m/s %.3g'%np.sqrt((u**2).mean()),'median|u| %.3g'%np.median(abs(u)),'p95 %.3g'%np.percentile(abs(u),95),'MSE(u/1e5) %.3g'%np.mean((u/1e5)**2))
print('RV rms V/m',float(RV.pow(2).mean().sqrt()),'elec_par loss',float(eq['elec_par'].pow(2).mean()))
print('Bohm share of nu_par: median %.2f'%float((p['alpha_B']*om/nu_full).median()))
print('Ver max m/s',float(Ver.abs().max())*p['V_ref'],'Vex range km/s',float(Vex.min())*p['V_ref']/1e3,float(Vex.max())*p['V_ref']/1e3)
print('elec_cont rms',float(eq['elec_cont'].pow(2).mean().sqrt()))
