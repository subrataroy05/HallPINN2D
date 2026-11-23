"""Read-only checkpoint evaluation, comparisons, figures and Gauss audit."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOCAL=ROOT/'experiments/nu_in_local'
CODE=HERE/'code'
REFERENCE_CODE=ROOT/'het_pinn_vfinal/experiments/bohm_momentum_ab/run_d/code'
REFERENCE_CKPT=REFERENCE_CODE.parent/'output/run_d_alpha1over200_vetheta2000_seed7_checkpoint.pt'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mplconfig')
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(CODE))
import mainScript as driver
import utils
from integrated_constraints import integrated_balances,axial_fluxes
torch.set_default_dtype(torch.float64)
KEYS=['poisson','ion_cont','neut_cont','elec_cont','energy','ion_mom_x','ion_mom_r',
      'elec_mom_x','elec_mom_r','elec_mom_theta','ion_cont_raw']

def module_from(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def load_original_utils():
    return module_from(REFERENCE_CODE/'utils.py','run_d_original_utils')

def load_model(path):
    c=torch.load(path,map_location='cpu',weights_only=False);d,p=c['domain'],c['physics']
    if 'JT_raw' in c['model_state_dict']:
        old=module_from(REFERENCE_CODE/'model.py','legacy_model')
        source=(REFERENCE_CODE/'mainScript.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='build_model')
        scope={'PINN':old.PINN};exec(ast.get_source_segment(source,node),scope)
        m=scope['build_model'](d,p,'cpu')
    else: m=driver.build_model(d,p,'cpu')
    m.load_state_dict(c['model_state_dict'],strict=True);m.eval();return m,d,p,c

def grid(d):
    xx,rr=torch.meshgrid(torch.linspace(d['x_min'],d['x_max'],200),torch.linspace(d['r_min'],d['r_max'],80),indexing='ij')
    return xx.reshape(-1),rr.reshape(-1)

def diagnostic(model,d,p,data):
    with torch.no_grad():
        x=torch.tensor([1.]);r=torch.full_like(x,.5*(d['r_min']+d['r_max']))
        v=float(model(x,r)[2])*p['V_ref']/1000
        currents=axial_fluxes(model,d,p,[0.,.5,1.])
        x,r=grid(d);f=model(x,r)
        l2=utils.data_loss_l2_relative(model,data)
        q=integrated_balances(model,d,p)
        # Same signs/denominators as Part A; include nonzero anode ion influx.
        vs=float(q['source_prefix'][-1]);gin=q['Gamma_in']
        bi=float(q['Gamma_i'][-1]-q['Gamma_i'][0]+q['wall_prefix'][-1])-vs
        bn=gin-float(q['Gamma_n'][-1])-vs
        return dict(Vix_exit_km_s=v,Id_A=currents['Id_A'].tolist(),ion_exit_A=float(currents['Gamma_i'][-1])*p['e_phys'],
                    grid_Vex_positive_fraction=float((f[6]>0).double().mean()),ni_pooled_L2=l2['ni'],
                    ion_imbalance_over_source=bi/vs,neutral_defect_over_inflow=bn/gin,
                    ion_training_residual_exit=float(q['ion_residual'][-1]),
                    neutral_training_residual_exit=float(q['neutral_residual'][-1]),
                    max_abs_Vex_normalized=float(f[6].abs().max()))

def measure(path,module=utils):
    before=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    m,d,p,c=load_model(path)
    data=module.load_haas_gallimore_data(str(CODE/'haas_gallimore_p5_5p4A.csv'),d,p,'cpu')
    assert all(len(v['values'])==145 for v in data.values())
    result=dict(checkpoint=str(path),sha256=before,epochs=c['epochs_completed'],final_loss=c['final_loss'],
                fit=module.data_loss_l2_relative(m,data),physics=p)
    with torch.no_grad():
        q=integrated_balances(m,d,p)
        result['integrated']={k:v.tolist() if torch.is_tensor(v) else v for k,v in q.items()}
        result['Id_stats_A']={k:float(v) for k,v in [('min',q['Id_A'].min()),('max',q['Id_A'].max()),('mean',q['Id_A'].mean())]}
        result['current']=dict(ion_A=float(q['Gamma_i'][-1])*p['e_phys'],electron_positive_e_integral_A=float(q['Gamma_e'][-1])*p['e_phys'],
                               electron_conventional_A=-float(q['Gamma_e'][-1])*p['e_phys'],net_A=float(q['Id_A'][-1]))
        x,r=grid(d);f=m(x,r);vex=f[6].reshape(200,80)
        result['current'].update(grid_Vex_positive_fraction=float((vex>0).double().mean()),exit_Vex_positive_fraction=float((vex[-1]>0).double().mean()))
        result['Vex_bound']=dict(bound_normalized=1000,max_abs_normalized=float(vex.abs().max()),max_abs_m_s=float(vex.abs().max())*p['V_ref'],
                                 fraction_above_95_percent=float((vex.abs()>950).double().mean()),fraction_above_99_percent=float((vex.abs()>990).double().mean()))
        rate=utils.compute_ion_neutral_frequency(f[4],f[2],p).numpy()
        result['nu_in_Hz']={k:float(v) for k,v in [('min',rate.min()),('median',np.median(rate)),('max',rate.max())]}
        xp=torch.tensor([0.,.25,.5,.75,1.]);fc=m(xp,torch.full_like(xp,.5*(d['r_min']+d['r_max'])))
        rates=utils.compute_ion_neutral_frequency(fc[4],fc[2],p)
        result['centerline']=[dict(x_hat=float(xx),Vix_km_s=float(v)*p['V_ref']/1000,nu_in_Hz=float(nu)) for xx,v,nu in zip(xp,fc[2],rates)]
        xc=torch.linspace(0,1,200);fc=m(xc,torch.full_like(xc,.5*(d['r_min']+d['r_max'])));peak=int(fc[1].argmax())
        result['ni_centerline_peak']=dict(normalized=float(fc[1][peak]),m3=float(fc[1][peak])*p['n_ref'],x_hat=float(xc[peak]),axial_grid_points=200)
        result['full_centerline']={'x_hat':xc.tolist(),'Vix_km_s':(fc[2]*p['V_ref']/1000).tolist()}
        radii=[];entry=data['ni']
        for rr in torch.unique(entry['coords'][:,1]):
            mask=entry['coords'][:,1]==rr;xx=entry['coords'][:,0][mask];yy=entry['values'][mask]
            pred=m(xx,entry['coords'][:,1][mask])[1];ex=torch.isclose(xx,torch.ones_like(xx),rtol=0,atol=1e-12)
            assert len(xx)==29 and int(ex.sum())==1
            value=float(yy[ex]);predicted=float(pred[ex])
            radii.append(dict(radius_mm=round(float(rr-d['r_min'])*p['L_canal']*1000,8),L2=float(torch.linalg.vector_norm(pred-yy)/torch.linalg.vector_norm(yy)),
                              exit_prediction_normalized=predicted,exit_data_normalized=value,exit_prediction_m3=predicted*p['n_ref'],exit_data_m3=value*p['n_ref']))
        result['ni_radii']=radii
    eq=module.compute_equation_residuals(m,x.clone().requires_grad_(True),r.clone().requires_grad_(True),p)
    result['residuals']={k:dict(MSE=float(eq[k].detach().square().mean()),RMS=float(eq[k].detach().square().mean().sqrt())) for k in KEYS}
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==before
    return result

def full_partA(path):
    import audit_partA as a
    a.load_model=load_model
    a.validate_shared_residuals(path,utils)
    rows=[]
    for nx,nr in [(200,64),(400,128),(800,256)]:
        row=a.evaluate(path,utils,nx,nr)
        if rows: row['change_vs_previous']=a.compare(rows[-1],row)
        rows.append(row)
        print(f'Final Part A audit {nx}x{nr}: ion={row["ion_imbalance_over_source"]:.9g}, neutral={row["neutral_defect_over_inflow"]:.9g}',flush=True)
    return dict(checkpoint=str(path),quadratures=rows,shared_residual_check='passed')

def generate_figures(path):
    m,d,p,c=load_model(path);out=HERE/'figures';out.mkdir(exist_ok=True)
    utils.create_output_folders(str(out))
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':13}): utils.save_centerline_profiles(m,d,p,str(out),'roy_current_seed7',nx=200,device='cpu')
    center=out/'Big_Plots/roy_current_seed7_centerline.png';center.replace(out/center.name)
    sys.path.insert(0,str(ROOT/'seed_cost_artifacts'))
    import make_errorbars
    original=make_errorbars.common.load_model
    try:
        make_errorbars.common.load_model=load_model
        with plt.rc_context(plt.rcParamsDefault): limits=make_errorbars.make_figures(path,out,True)
    finally: make_errorbars.common.load_model=original
    return dict(paths=[str(out/center.name)]+[str(out/f'ch5_hg_{q}_5radii.png') for q in ('ni','Te','Vp')],axis_limits=limits,visual_inspection='pending')
