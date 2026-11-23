"""frozenF_tensor_fli_local (Roy 3 Nov 13:07): warm start from frozenD_mobility (last frozen-heavy checkpoint); anisotropic transport tensor; field-line-integral constraint as principal new term (M_ref 3 V, ramp 2500 epochs, weights: 1 downstream of 20 mm, smooth 15-20 mm, 0.1 at 10-15 mm, none below 10 mm where lines are truncated); retains a weak current-equivalent local residual R_par/R_allow (J_allow = 47.4 A/m^2, 10% of I_d/A); data weighting and adaptive 5.4 A penalty unchanged; 8000 epochs. Previous doc: frozenE_tensor_fli (Roy 3 Nov 11:33): electron-only, heavy species frozen from stage C, warm start from frozenC_electron; field-aligned transport tensor (nu_par classical, nu_perp with Bohm); local parallel residuals switched off (elec_par=0, elec_par_mob=0); field-line-integral constraint Delta phi - int dp_e/n_e on traced field lines, full weight for lines seeded downstream of x=20 mm, 0.1 upstream; data weighting and adaptive 5.4 A current penalty as in frozenD. Previous doc: frozenD_mobility (Roy 2 Nov 22:03): electron-only continuation from frozenC_electron, heavy species frozen from stage C; adds the mobility-weighted parallel residual (cosine ramp 0->1 over 3000 epochs), keeps elec_par=10, starts the 5.4 A integrated-current penalty at a moderate 10 and raises it x3 only if data and electron continuity stay stable. Earlier doc: bfield_2d with widened potential correction envelope (2A*tanh(raw/2), A=25): removes the cap that made 75/145 Vp points unreachable, warm start from bfield_2d final. Original doc: Warm start from ion_ramp with hard anode phi=274.04 V. Base physics, network and loss functions are reused unchanged."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import analyze as a
import diagnostics as D
from integrated_constraints import quadrature

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'stageC_current/output/checkpoint.pt'
OUT = HERE/'output'
DATA_WEIGHTS = {'ni':1., 'Te':1., 'Vp':1.}  # uncertainty-weighted (chi^2/N) data terms
REQUESTED_EPOCHS = 6000
STOP_REQUEST = None

def now():
    return datetime.now(timezone.utc).isoformat()

def request_stop(signum, frame):
    global STOP_REQUEST
    STOP_REQUEST = f'user/process signal {signal.Signals(signum).name}'

PAR_TARGET=0.; PAR_RAMP=1  # local parallel residual off (Roy 3 Nov)
MOB_TARGET=0.; MOB_RAMP=1  # mobility-weighted local residual off
FLI_TARGET=float(os.environ.get('FLI_TARGET','1.0')); FLI_RAMP=2500; FLI_UP_WEIGHT=0.1; FLI_SPLIT_MM=20.0; FLI_TRANS_MM=15.0; FLI_CUT_MM=10.0; FLI_NLINES=24; FLI_K=64; FLI_MREF_V=3.0
LOCAL_J_W=float(os.environ.get('LOCAL_J_W','0')); EPOCHS=8000
CUR_LEVELS=[10.,30.,100.]; CUR_STATE={'level':0}
WARM = HERE.parent/'frozenD_mobility/output/checkpoint.pt'
FLI={}
CUR_TARGET=100.; CUR_RAMP=1  # integrated discharge-current condition (I_d(x)=5.4 A at 40 stations), cosine ramp
def cur_weight(epoch):
    return CUR_LEVELS[CUR_STATE['level']]

def fli_weight(epoch):
    s=min(1.,epoch/FLI_RAMP)
    return FLI_TARGET*(1-math.cos(math.pi*s))/2

def mob_weight(epoch):
    s=min(1.,epoch/MOB_RAMP)
    return MOB_TARGET*(1-math.cos(math.pi*s))/2

def par_weight(epoch):
    s=min(1.,epoch/PAR_RAMP)
    return PAR_TARGET*(1-math.cos(math.pi*s))/2

def ion_weight(epoch):
    return 1000.

def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)

def save_torch(path, obj):
    path = Path(path); path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    temp.replace(path)

def verify_bases():
    manifest = json.loads((HERE/'base_sha256.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())

class FrozenHeavy(torch.nn.Module):
    """Roy 2 Nov test 3: ion and neutral fields (n_i, V_ix, V_ir, n_n) frozen from stage C;
    phi, eta (n_e), T_e and the electron velocities trainable. n_e = n_i_frozen (1 + eta)."""
    def __init__(self, train, frozen):
        super().__init__()
        self.train_net = train
        object.__setattr__(self, 'frozen', frozen)
        for q in frozen.parameters(): q.requires_grad_(False)
    def __getattr__(self, name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(self._modules['train_net'], name)
    def forward(self, x, r):
        phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vet, eta = self.train_net(x, r)
        _, niF, VixF, VirF, nnF, _, _, _, _, _, _ = self.frozen(x, r)
        return phi, niF, VixF, VirF, nnF, niF*(1+eta), Vex, Ver, Te, Vet, eta

def build_field_lines(d,p):
    """Trace FLI_NLINES field lines of the measured field through mid-radius seeds, both directions, to the domain boundary;
    resample each to FLI_K points by arc length. Fixed geometry (heavy fields and B are frozen)."""
    from field import B as Bf
    rlo,rhi=d['r_min'],d['r_max']; rm=.5*(rlo+rhi); ds=2e-3
    seeds=np.linspace(FLI_CUT_MM/(p['L_canal']*1e3)+0.0125,0.9875,FLI_NLINES)
    def bdir(x,r):
        bx,br=Bf(torch.tensor(x),torch.tensor(r)); n=torch.hypot(bx,br); return (bx/n).numpy(),(br/n).numpy()
    paths={}
    for sgn in (-1,1):
        x=seeds.copy(); r=np.full_like(x,rm); act=np.ones(len(x),bool); P=[[(a,b)] for a,b in zip(x,r)]
        for _ in range(5000):
            if not act.any(): break
            f=lambda x,r: tuple(sgn*v for v in bdir(x,r))
            k1=f(x,r);k2=f(x+ds/2*k1[0],r+ds/2*k1[1]);k3=f(x+ds/2*k2[0],r+ds/2*k2[1]);k4=f(x+ds*k3[0],r+ds*k3[1])
            xn=x+ds/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);rn=r+ds/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            act=act&(xn>=FLI_CUT_MM/(p['L_canal']*1e3))&(xn<=1)&(rn>=rlo)&(rn<=rhi)
            for k in np.where(act)[0]: P[k].append((xn[k],rn[k]))
            x=np.where(act,xn,x);r=np.where(act,rn,r)
        paths[sgn]=P
    XS=[];RS=[];W=[];meta=[]
    for k,s0 in enumerate(seeds):
        Q=np.array(paths[-1][k][::-1]+paths[1][k][1:]); sl=np.r_[0,np.cumsum(np.hypot(*np.diff(Q,axis=0).T))]
        t=np.linspace(0,sl[-1],FLI_K); XS.append(np.interp(t,sl,Q[:,0])); RS.append(np.interp(t,sl,Q[:,1]))
        xm=s0*p['L_canal']*1e3; t_=min(1.,max(0.,(xm-FLI_TRANS_MM)/(FLI_SPLIT_MM-FLI_TRANS_MM))); W.append(FLI_UP_WEIGHT+(1-FLI_UP_WEIGHT)*(1-math.cos(math.pi*t_))/2)
        meta.append(dict(seed_x_hat=float(s0),length_mm=float(sl[-1]*p['L_canal']*1e3),start=[float(Q[0,0]),float(Q[0,1])],end=[float(Q[-1,0]),float(Q[-1,1])],weight=W[-1]))
    FLI.update(x=torch.tensor(np.array(XS)),r=torch.tensor(np.array(RS)),w=torch.tensor(W),meta=meta)
    write_json(HERE/'field_lines.json',meta)

def fli_mismatch_V(model,p):
    """Delta phi - int d(n_e T_e)/n_e along each traced line, in volts (Boltzmann with variable T_e => 0)."""
    nl,K=FLI['x'].shape; f=model(FLI['x'].reshape(-1),FLI['r'].reshape(-1))
    phi=f[0].reshape(nl,K); ne=f[5].reshape(nl,K); te=f[8].reshape(nl,K); pe=ne*te
    M=(phi[:,-1]-phi[:,0])-((pe[:,1:]-pe[:,:-1])/(0.5*(ne[:,1:]+ne[:,:-1]))).sum(1)
    return M*p['phi_ref']

def setup():
    torch.set_num_threads(4)
    torch.manual_seed(7); np.random.seed(7)
    model, domain, physics, original = a.load_model(BASE)
    domain = dict(domain)
    domain["phi_left"] = 274.04 / physics["phi_ref"]
    model.phi_left = domain["phi_left"]
    physics = dict(physics)
    physics["magnetic_field_model"] = "flux_2d"
    physics["electron_pressure"] = True
    model.train()
    weights = dict(original['weights']); weights['integrated_current'] = 0.
    assert original['weights']['integrated_ion']==1000
    assert original['weights']['integrated_neutral']==100
    assert all(weights[k]==original['weights'][k] for k in weights if k!='integrated_current')
    assert 'exit_ion_current' not in weights
    weights['elec_par'] = PAR_TARGET
    import copy
    frozen = copy.deepcopy(model); frozen.eval()
    model = FrozenHeavy(model, frozen)
    warm=torch.load(WARM,map_location='cpu',weights_only=False)
    model.load_state_dict(warm['model_state_dict'],strict=True)
    physics['V_par_ref']=1.0e6
    weights['elec_par_mob']=0.
    physics['transport_tensor']=True
    weights['fli']=0.
    weights['elec_par_J']=LOCAL_J_W
    physics['J_allow']=47.4
    build_field_lines(domain,physics)
    for k in ['ion_cont','neut_cont','ion_mom_x','ion_mom_r','integrated_ion','integrated_neutral','Vir_wall','smooth_ni','smooth_nn']:
        if k in weights: weights[k] = 0.
    assert physics['nn_inlet']==.164145
    torch.manual_seed(2601); np.random.seed(2601)
    x, r = a.utils.sample_domain_points(10000,domain,'cpu')
    boundary = a.utils.sample_boundary_points(1000,1000,domain,'cpu')
    data = a.utils.load_haas_gallimore_data(str(HERE/'code/haas_gallimore_p5_5p4A.csv'),domain,physics,'cpu')
    assert all(len(v['values'])==145 for v in data.values())
    optimizer = torch.optim.Adam([q for q in model.parameters() if q.requires_grad],lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=300,min_lr=1e-6)
    return model,domain,physics,original,weights,x,r,boundary,data,optimizer,scheduler

def uncertainty_data_loss(model,data,physics):
    """Roy 1 Nov: data residuals normalized by measurement uncertainty, mean of z^2 per field.
    n_i: +-50% of measured value; T_e: +-20%; V_p: asymmetric -3/+6 V (model above data uses 6 V)."""
    out={}
    for q,idx in {'ni':1,'Te':8,'Vp':0}.items():
        c=data[q]['coords']; obs=data[q]['values']; pred=model(c[:,0],c[:,1])[idx]
        if q=='ni': z=(pred-obs)/(0.5*obs)
        elif q=='Te': z=(pred-obs)/(0.2*obs)
        else:
            e=(pred-obs)*physics['phi_ref']; z=torch.where(e>0,e/6.0,e/3.0)
        out[q]=(z*z).mean()
    return out

def evaluate_loss(model,domain,physics,weights,x,r,boundary,data):
    total, raw = a.utils.compute_all_losses(model,x,r,boundary,domain,physics,weights)
    dl = uncertainty_data_loss(model,data,physics)
    weighted_data = DATA_WEIGHTS['ni']*dl['ni'] + DATA_WEIGHTS['Te']*dl['Te'] + DATA_WEIGHTS['Vp']*dl['Vp']
    total = total + weighted_data
    Mv = fli_mismatch_V(model,physics); fli_loss = (FLI['w']*(Mv/FLI_MREF_V)**2).sum()/FLI['w'].sum()
    total = total + weights.get('fli',0.)*fli_loss; raw['fli'] = float(fli_loss.detach())
    for field,w in DATA_WEIGHTS.items():
        raw['data_'+field] = float(dl[field].detach())
    combined = dict(weights,**{'data_'+k:v for k,v in DATA_WEIGHTS.items()})
    weighted = {k:w*raw[k] for k,w in combined.items()}
    value = float(total.detach())
    assert math.isclose(sum(weighted.values()),value,rel_tol=1e-10,abs_tol=1e-9) or not math.isfinite(value)
    return total, {'total':value,'raw':{k:raw[k] for k in combined},'weighted':weighted,
                   'weights':combined,'shares_percent':{k:100*v/value if value else 0. for k,v in weighted.items()}}

def append_loss(epoch, loss, optimizer, preupdate_total=None):
    row = dict(epoch=epoch, phase='post_update' if epoch else 'initial',
               ion_weight=ion_weight(epoch), total=loss['total'],
               lr=optimizer.param_groups[0]['lr'], preupdate_total=preupdate_total, timestamp_utc=now())
    for k in loss['raw']:
        row.update({k+'_raw':loss['raw'][k],k+'_weighted':loss['weighted'][k],k+'_share_pct':loss['shares_percent'][k]})
    path=HERE/'loss_terms.csv'; new=not path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if new: writer.writeheader()
        writer.writerow(row);f.flush();os.fsync(f.fileno())
    return row

@torch.no_grad()
def diagnostic(model,d,p,data):
    q=a.integrated_balances(model,d,p)
    current=a.axial_fluxes(model,d,p,[0.,.5,1.])
    _,radius,wr,xs,wx=quadrature(d['x_min'],d['x_max'],d['r_min'],d['r_max'])
    walls=[]
    for rr,sign in [(d['r_min'],-1),(d['r_max'],1)]:
        xt=torch.tensor(xs.reshape(-1)); f=model(xt,torch.full_like(xt,rr))
        area=torch.tensor(sign*2*np.pi*rr*p['L_canal']**2*wx)
        walls.append(float(((f[1]*f[3]).reshape(xs.shape)*area).sum())*p['n_ref']*p['V_ref'])
    gin=float(q['Gamma_in']); source=float(q['source_prefix'][-1]);e=p['e_phys']
    gi0=float(q['Gamma_i'][0]);gi1=float(q['Gamma_i'][-1]);gn1=float(q['Gamma_n'][-1])
    idefect=gi1-gi0+sum(walls)-source;ndefect=gin-gn1-source
    b=dict(ion_anode_in_s=gi0,ion_exit_out_s=gi1,ion_inner_wall_out_s=walls[0],ion_outer_wall_out_s=walls[1],
           source_s=source,ion_defect_s=idefect,neutral_in_s=gin,neutral_out_s=gn1,neutral_defect_s=ndefect)
    b.update({k[:-2]+'_Aeq':v*e for k,v in list(b.items())})
    b.update(ion_imbalance_over_source=idefect/source if source else float('nan'),
             neutral_defect_over_inflow=ndefect/gin,source_mass_utilization=source/gin,
             exit_ion_over_neutral_inflow=gi1/gin)
    xr,rr=a.grid(d);f=model(xr,rr)
    fields={name:dict(min=float(f[i].min())*scale,max=float(f[i].max())*scale)
            for name,i,scale in [('ni_m3',1,p['n_ref']),('nn_m3',4,p['n_ref']),('ne_m3',5,p['n_ref']),
                                  ('Vix_m_s',2,p['V_ref']),('Vex_m_s',6,p['V_ref'])]}
    finite=all(bool(torch.isfinite(v).all()) for v in f)
    positive=all(bool((f[i]>0).all()) for i in (1,4,5))
    maxvex=float(f[6].abs().max())
    fr=model(torch.ones(len(radius)),torch.tensor(radius));area=torch.tensor(2*np.pi*radius*wr*p['L_canal']**2)
    meanv=float((fr[1]*fr[2]*area).sum()/(fr[1]*area).sum())*p['V_ref']/1000
    center=float(model(torch.tensor([1.]),torch.tensor([.5*(d['r_min']+d['r_max'])]))[2])*p['V_ref']/1000
    coverage={}
    for field,idx in a.utils.DATA_OUTPUT_INDICES.items():
        entry=data[field];obs=entry['values'];coord=entry['coords'];pred=model(coord[:,0],coord[:,1])[idx]
        lo,hi=(3/12.1,6/12.1) if field=='Vp' else ((.5 if field=='ni' else .2)*obs,)*2
        coverage[field]=float(((pred>=obs-lo)&(pred<=obs+hi)).double().mean())
    iexit=gi1*e;idexit=float(q['Id_A'][-1])
    with torch.enable_grad():
        dd=D.sections(model,d,p,np.linspace(0,1,40),nr=128,force_2d=True)
        xg,rg=a.grid(d);xg=xg.clone().requires_grad_(True);rg=rg.clone().requires_grad_(True)
        ec=a.utils.compute_equation_residuals(model,xg,rg,p);ecr=float(ec['elec_cont'].detach().pow(2).mean().sqrt())
        mob=float(ec['elec_par_mob'].detach().pow(2).mean());locJ=float(ec['elec_par_J'].detach().pow(2).mean());Ral=float(ec['R_allow'].detach().median());rpar=float((ec['elec_par'].detach()*p.get('E_ref_par',1200.)).pow(2).mean().sqrt())
    ewall=[]
    for rr_,sign in [(d['r_min'],-1),(d['r_max'],1)]:
        xt=torch.tensor(xs.reshape(-1)); f_=model(xt,torch.full_like(xt,rr_))
        area_=torch.tensor(sign*2*np.pi*rr_*p['L_canal']**2*wx)
        ewall.append(float(((f_[5]*f_[7]).reshape(xs.shape)*area_).sum())*p['n_ref']*p['V_ref']*e)
    extra=dict(I_DD_40_A=[r_['Id_DD_A'] for r_ in dd],I_DD_par_40_A=[r_['electron_parallel_conventional_A'] for r_ in dd],
               fli_mismatch_V=[float(v) for v in fli_mismatch_V(model,p)],fli_seed_x_hat=[m_['seed_x_hat'] for m_ in FLI['meta']],
               elec_par_J_mse=locJ,R_allow_median_V_m=Ral,fli_weights=[float(v) for v in FLI['w']],
               elec_cont_rms=ecr,elec_par_mob_mse=mob,Rpar_rms_grid_V_m=rpar,electron_wall_flux_Aeq=ewall,current_penalty=CUR_LEVELS[CUR_STATE['level']])
    return dict(balance=b,fields=fields,all_fields_finite=finite,densities_positive=positive,
                max_abs_Vex_normalized=maxvex,grid_Vex_positive_fraction=float((f[6]>0).double().mean()),
                exit_Vex_positive_fraction=float((f[6].reshape(200,80)[-1]>0).double().mean()),
                Id_at_0_half_1_A=current['Id_A'].tolist(),Id_40_A=q['Id_A'].tolist(),
                Id_min_A=float(q['Id_A'].min()),Id_max_A=float(q['Id_A'].max()),
                exit_ion_A=iexit,exit_electron_conventional_A=-float(q['Gamma_e'][-1])*e,
                Ii_over_Id_exit=iexit/idexit if idexit else None,
                Vix_exit_centerline_km_s=center,Vix_exit_density_weighted_mean_km_s=meanv,
                pooled_L2=a.utils.data_loss_l2_relative(model,data),band_fractions=coverage,**extra)

def physical_stops(d):
    reasons=[]
    if not d['all_fields_finite'] or not d['densities_positive']:reasons.append('nonfinite field or nonpositive density')
    if d['fields']['ni_m3']['max']>6e18:reasons.append('ni exceeds 6e18 m^-3')
    if not 2<=d['Vix_exit_centerline_km_s']<=20:reasons.append('centerline exit Vix outside [2,20] km/s')
    if d['max_abs_Vex_normalized']>950:reasons.append('|Vex| exceeds 95% of bound')
    if abs(d['balance']['neutral_defect_over_inflow'])>.30:reasons.append('|neutral imbalance| exceeds 30%')
    return reasons

def append_diagnostic(epoch,d):
    entry=dict(epoch=epoch,timestamp_utc=now(),**d)
    with (HERE/'progress_diagnostics.jsonl').open('a') as f:
        f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
    from reporting import diagnostic_markdown
    with (HERE/'progress.md').open('a') as f:
        f.write(f'\n## Epoch {epoch}\n\n'+diagnostic_markdown(d)+'\n');f.flush();os.fsync(f.fileno())
    print('[diagnostic] '+json.dumps(entry),flush=True)

def snapshot(model,d,p,original,weights,optimizer,scheduler,epoch,loss,reason,elapsed):
    return dict(model_state_dict=model.state_dict(),domain=d,physics=p,weights=dict(weights),
                optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),
                epochs_completed=epoch,base_epochs_completed=original['epochs_completed'],init_seed=7,collocation_seed=2601,
                run_name='frozenC_electron_seed7',eta_max=float(model.eta_max),final_loss=loss['total'],
                loss_at_saved_weights=loss,loss_epoch=epoch if loss.get('at_epoch')==epoch else loss.get('at_epoch'),
                loss_matches_saved_weights=loss.get('at_epoch')==epoch,
                optimizer_name='Adam',optimizer_initial_lr=2e-5,fresh_optimizer_state_at_start=True,
                resume_from=str(BASE),stop_reason=reason,elapsed_loop_seconds=elapsed,
                data_anneal={'final_weights':DATA_WEIGHTS},saved_utc=now())

def finish():
    from reporting import refresh
    path=OUT/'checkpoint.pt'
    write_json(HERE/'final_metrics.json',a.measure(path))
    refresh()
    write_json(HERE/'final_partA.json',a.full_partA(path))
    write_json(HERE/'figures.json',a.generate_figures(path))
    verify_bases()
    state=json.loads((HERE/'progress.json').read_text())
    state.update(postprocessing='complete',base_files_unchanged=True)
    write_json(HERE/'progress.json',state);refresh()

def train():
    from reporting import refresh
    assert json.loads((HERE/'preflight.json').read_text())['passed']
    assert json.loads((HERE/'regression.json').read_text())['passed']
    assert not (HERE/'loss_terms.csv').exists(),'Refusing to overwrite a previous run.'
    verify_bases();OUT.mkdir(exist_ok=True)
    model,d,p,original,weights,x,r,boundary,data,opt,scheduler=setup()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,request_stop)
    state=dict(status='initializing',epochs_completed=0,requested_epochs=EPOCHS,pid=os.getpid(),started_utc=now(),threads=4)
    (HERE/'pid.txt').write_text(str(os.getpid())+'\n')
    (HERE/'progress.md').write_text('# frozenF_tensor_fli_local progress\n\nPost-update diagnostics every 500 epochs. '
        'Physical balances use the 40-station, GL32 radial / GL8-per-interval training quadrature. '
        'Inner/outer wall fluxes are signed outward; anode entry is positive inward. '
        'All ranges and sign fractions use the inclusive 200×80 grid.\n')
    epoch=0;start=time.perf_counter();reason='initializing';loss=None;last_pre=None
    logged_epochs=set();diag_epochs=set();bad_ion=bad_loss=0
    try:
        total,loss=evaluate_loss(model,d,p,weights,x,r,boundary,data);del total
        loss['at_epoch']=0;initial_loss=loss['total']
        append_loss(0,loss,opt);logged_epochs.add(0)
        save_torch(OUT/'checkpoint_epoch_0000.pt',snapshot(model,d,p,original,weights,opt,scheduler,0,loss,'initial',0))
        baseline=diagnostic(model,d,p,data);append_diagnostic(0,baseline);diag_epochs.add(0)
        write_json(HERE/'baseline_diagnostic.json',baseline)
        baseline_ion=abs(baseline['balance']['ion_imbalance_over_source'])
        if physical_stops(baseline):print('[warning] epoch-0 physical flags (not a stop): '+'; '.join(physical_stops(baseline)),flush=True)
        state.update(status='training',epoch0_total=initial_loss,loop_started_utc=now())
        write_json(HERE/'progress.json',state);refresh()
        start=time.perf_counter()
        reason=f'completed {EPOCHS} epochs'
        for step in range(1,EPOCHS+1):
            if STOP_REQUEST:reason=STOP_REQUEST;break
            weights['elec_par']=par_weight(step); weights['integrated_current']=cur_weight(step); weights['elec_par_mob']=mob_weight(step); weights['fli']=fli_weight(step)
            opt.zero_grad()
            total,pre=evaluate_loss(model,d,p,weights,x,r,boundary,data)
            last_pre=pre['total']
            if not math.isfinite(last_pre):
                reason='nonfinite training loss; immediate safety stop before optimizer update';del total;break
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            if step>FLI_RAMP: scheduler.step(last_pre)  # no LR decay while the parallel weight is ramping (total loss rises by construction)
            epoch=step
            del total
            if epoch%100==0:
                value,loss=evaluate_loss(model,d,p,weights,x,r,boundary,data);del value
                loss['at_epoch']=epoch;append_loss(epoch,loss,opt,last_pre);logged_epochs.add(epoch)
                if not math.isfinite(loss['total']):reason='nonfinite post-update loss';break
                print(f'epoch {epoch} total={loss["total"]:.12g} ion_weight={weights["integrated_ion"]:g} lr={opt.param_groups[0]["lr"]:.6g}',flush=True)
            if epoch%500==0:
                save_torch(OUT/f'checkpoint_epoch_{epoch:04d}.pt',snapshot(model,d,p,original,weights,opt,scheduler,epoch,loss,'block complete',time.perf_counter()-start))
                diag=diagnostic(model,d,p,data);append_diagnostic(epoch,diag);diag_epochs.add(epoch)
                if epoch>=2000 and epoch%1000==0 and CUR_STATE['level']<len(CUR_LEVELS)-1:
                    ok=(diag['band_fractions']['Vp']>=baseline['band_fractions']['Vp']-.03 and diag['band_fractions']['Te']>=baseline['band_fractions']['Te']-.03
                        and diag['band_fractions']['ni']>=baseline['band_fractions']['ni']-.03 and diag['elec_cont_rms']<=1.5*baseline['elec_cont_rms'])
                    if ok: CUR_STATE['level']+=1
                    print(f'[current penalty] epoch {epoch} stable={ok} -> weight {CUR_LEVELS[CUR_STATE["level"]]:g}',flush=True)
                bad_ion=bad_ion+1 if abs(diag['balance']['ion_imbalance_over_source'])>baseline_ion else 0
                bad_loss=bad_loss+1 if loss['total']>10*initial_loss else 0
                reasons=physical_stops(diag)
                if bad_ion>=3:reasons.append('ion imbalance worse than epoch 0 at three consecutive checks')
                if bad_loss>=2:reasons.append('total loss >10x epoch 0 at two consecutive checks')
                state.update(latest_diagnostic_epoch=epoch,consecutive_worse_ion=bad_ion,consecutive_excess_loss=bad_loss)
                # Only nonfinite losses stop this run; inherited physical stops are logged as warnings.
                state['physical_flags']='; '.join(reasons)
                if reasons:print(f'[warning] epoch {epoch} physical flags (not a stop): '+state['physical_flags'],flush=True)
            if epoch%25==0:
                elapsed=time.perf_counter()-start
                state.update(epochs_completed=epoch,elapsed_loop_seconds=elapsed,seconds_per_epoch=elapsed/epoch,
                             last_preupdate_loss=last_pre,lr=opt.param_groups[0]['lr'],ion_weight=weights['integrated_ion'],
                             projected_end_local=(datetime.now().astimezone()+timedelta(seconds=elapsed/epoch*(EPOCHS-epoch))).isoformat())
                write_json(HERE/'progress.json',state)
                if epoch%500==0:refresh()
        if STOP_REQUEST:reason=STOP_REQUEST
    except BaseException as exc:
        reason=f'exception: {type(exc).__name__}: {exc}'
        state['exception']=reason
        import traceback
        state['traceback']=traceback.format_exc()
        print(state['traceback'],flush=True)
    finally:
        # Persist weights FIRST, before any potentially failing evaluation/report work.
        elapsed=time.perf_counter()-start
        if loss is None: loss={'total':float('nan'),'at_epoch':None}
        final=snapshot(model,d,p,original,weights,opt,scheduler,epoch,loss,reason,elapsed)
        save_torch(OUT/'checkpoint.pt',final)
        save_torch(OUT/f'checkpoint_stop_{epoch:04d}.pt',final)
        try:
            if epoch not in logged_epochs:
                value,loss=evaluate_loss(model,d,p,weights,x,r,boundary,data);del value
                loss['at_epoch']=epoch;append_loss(epoch,loss,opt,last_pre)
                final=snapshot(model,d,p,original,weights,opt,scheduler,epoch,loss,reason,elapsed)
                save_torch(OUT/'checkpoint.pt',final);save_torch(OUT/f'checkpoint_stop_{epoch:04d}.pt',final)
            if epoch not in diag_epochs:append_diagnostic(epoch,diagnostic(model,d,p,data))
        except BaseException as exc:
            state['stop_evaluation_error']=repr(exc)
        state.update(status='complete' if epoch==EPOCHS else 'stopped',epochs_completed=epoch,stop_reason=reason,
                     training_wall_seconds=elapsed,ended_utc=now(),checkpoint=str(OUT/'checkpoint.pt'),postprocessing='pending')
        state.pop('projected_end_local',None)
        write_json(HERE/'progress.json',state);refresh()
    try:raise RuntimeError('postprocessing skipped for frozen-heavy wrapper')
    except BaseException as exc:
        state=json.loads((HERE/'progress.json').read_text());state['postprocessing_error']=repr(exc)
        write_json(HERE/'progress.json',state);refresh();raise

if __name__=='__main__':
    train()
