"""Audit of Run D (reference) for C1-C3: a posteriori magnitudes, no retraining."""
import sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'het_pinn_vfinal/experiments/bohm_momentum_ab/run_d/code'
sys.path.insert(0,str(CODE))
import numpy as np, torch
import mainScript as M, utils as U
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
ck=torch.load(CODE.parent/'output/run_d_alpha1over200_vetheta2000_seed7_checkpoint.pt',map_location='cpu',weights_only=False)
d,p=ck['domain'],ck['physics']; m=M.build_model(d,p,'cpu'); m.load_state_dict(ck['model_state_dict']); m.eval()
xx,rr=torch.meshgrid(torch.linspace(d['x_min'],d['x_max'],200),torch.linspace(d['r_min'],d['r_max'],80),indexing='ij')
x=xx.reshape(-1).clone().requires_grad_(True); r=rr.reshape(-1).clone().requires_grad_(True)
eq=U.compute_equation_residuals(m,x,r,p)
phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=eq['fields']
g=lambda y,z: torch.autograd.grad(y.sum(),z,retain_graph=True)[0]
Si=U.compute_Si(nn_,ne,Te,p); coll=U.compute_collision_frequencies(ne,nn_,Te,p)
tau=p.get('tau_ref',p['L_canal']/p['V_ref']); mu=p['me']/p['mi']
rms=lambda t: float(t.detach().pow(2).mean().sqrt())
es=p.get('energy_scale',1.0)
ion_term=Si*(1.5*Te+p.get('energy_alpha',1.0)*p.get('eps_i',1.0))
RE=eq['energy']; RE_fixed=RE+2*ion_term/es
out={}
out['C1']=dict(energy_res_rms_as_trained=rms(RE),energy_res_rms_sign_corrected_same_fields=rms(RE_fixed),
  ionization_term_rms=rms(ion_term/es),ohmic_rms=rms((ne*Vex*g(phi,x)+ne*Ver*g(phi,r))/es),
  enthalpy_flux_rms=rms((g(ne*Vex*2.5*Te,x)+g(ne*Ver*2.5*Te,r)+ne*Ver*2.5*Te/r)/es))
drag_all=ni*mu*coll['nu_total']*tau*(Vex-Vix); drag_ei=ni*mu*coll['nu_ei']*tau*(Vex-Vix)
ims=p.get('ion_mom_scale',1.0)
out['C2']=dict(drag_nu_ei_plus_en_rms=rms(drag_all/ims),drag_nu_ei_only_rms=rms(drag_ei/ims),
  difference_rms=rms((drag_all-drag_ei)/ims),ion_mom_x_res_rms=rms(eq['ion_mom_x']),electric_rms=rms(ni*g(phi,x)/ims))
cent=mu*ne*Vet**2/r; cor=mu*ne*Ver*Vet/r
nuh=mu*(coll['nu_total']+p['alpha_B']*p['e_over_me']*U.compute_Br(x,p) if hasattr(U,'compute_Br') else coll['nu_total'])*tau
out['C3']=dict(centrifugal_rms=rms(cent),centrifugal_max=float(cent.detach().abs().max()),
  radial_electric_rms=rms(ne*g(phi,r)),elec_mom_r_res_rms=rms(eq['elec_mom_r']),
  coriolis_rms=rms(cor),elec_mom_theta_res_rms=rms(eq['elec_mom_theta']),
  hall_theta_rms=rms(eq['elec_mom_theta']*0+mu*p['e_over_me']*tau*U.compute_Br(x,p)*ne*Vex) if hasattr(U,'compute_Br') else None,
  Vet_max=float(Vet.detach().abs().max()))
print(json.dumps(out,indent=1)); json.dump(out,open(Path(__file__).parent/'audit_rund.json','w'),indent=1)
