"""Electron-only network on detached, spatially differentiated frozen coefficients."""
from common import *
import csv
import signal
import time
import traceback

LABELS=('electron_continuity','electron_momentum_x','electron_momentum_r','electron_momentum_theta','anode_current')
STOP=None

def stop_signal(sig,frame):
    global STOP
    STOP=f'external {signal.Signals(sig).name}'

class ElectronNet(torch.nn.Module):
    def __init__(self,d):
        super().__init__();self.d=dict(d)
        layers=[torch.nn.Linear(2,64),torch.nn.Tanh()]
        for _ in range(3):layers.extend([torch.nn.Linear(64,64),torch.nn.Tanh()])
        layers.append(torch.nn.Linear(64,3));self.net=torch.nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer,torch.nn.Linear):torch.nn.init.xavier_uniform_(layer.weight);torch.nn.init.zeros_(layer.bias)

    def forward(self,x,r):
        u=self.net(torch.stack((x,r),dim=1))
        xi=(x-self.d['x_min'])/(self.d['x_max']-self.d['x_min'])
        rho=(r-self.d['r_min'])/(self.d['r_max']-self.d['r_min'])
        return 1000*torch.tanh(u[:,0]),.008*rho*(1-rho)*torch.tanh(u[:,1]),8000*xi*(1-xi)*torch.tanh(u[:,2])

class Composite(torch.nn.Module):
    """Evaluation only: enable spatial derivatives, never background parameter gradients."""
    def __init__(self,bg,net):
        super().__init__();self.bg=bg;self.electrons=net;self.eta_max=bg.eta_max
    def forward(self,x,r):
        f=list(self.bg(x,r));f[6],f[7],f[9]=self.electrons(x,r);return tuple(f)

def load_background():
    bg,d,p,c=a.load_model(BASE)
    for par in bg.parameters():par.requires_grad_(False)
    bg.eval();return bg,d,p,c

def coefficients(bg,p,x,r):
    """Compute spatial derivatives once, then detach all background quantities."""
    xx=x.detach().clone().requires_grad_(True);rr=r.detach().clone().requires_grad_(True)
    f=bg(xx,rr);phi,ni,vi,vir,nn,ne,_,_,te,_,eta=f
    coll=a.utils.compute_collision_frequencies(ne,nn,te,p)
    bx,br=a.utils.compute_B_components(xx,rr,p)
    omega=p['e_over_me']*torch.sqrt(bx.square()+br.square())
    nu=coll['nu_total']+p['alpha_B']*omega
    pressure=ne*te
    values=dict(phi=phi,ni=ni,Vix=vi,Vir=vir,nn=nn,ne=ne,Te=te,eta=eta,
                ne_x=grad(ne,xx),ne_r=grad(ne,rr),phi_x=grad(phi,xx),phi_r=grad(phi,rr),
                pressure_x=grad(pressure,xx),pressure_r=grad(pressure,rr),
                Si=a.utils.compute_Si(nn,ne,te,p),omega_tau=omega*p['tau_ref'],nu_tau=nu*p['tau_ref'],omega_x_tau=p['e_over_me']*bx*p['tau_ref'],omega_r_tau=p['e_over_me']*br*p['tau_ref'])
    return {k:v.detach().clone() for k,v in values.items()}

def pressure_coefficient(p):
    # T_ref is 12.1 eV; convert eV to joules using the code's elementary charge.
    Tref_eV=p['phi_ref']
    return p['e_phys']*Tref_eV/(p['mi']*p['V_ref']**2)

def equations(net,x,r,b,p,pressure):
    vx,vr,vt=net(x,r)
    vx_x=grad(vx,x);vx_r=grad(vx,r);vr_x=grad(vr,x);vr_r=grad(vr,r);vt_x=grad(vt,x);vt_r=grad(vt,r)
    ne=b['ne'];mu=p['me']/p['mi'];chi=pressure_coefficient(p) if pressure else 0.
    cont=b['ne_x']*vx+ne*vx_x+b['ne_r']*vr+ne*vr_r+ne*vr/r-b['Si']
    tx=dict(inertia=mu*ne*(vx*vx_x+vr*vx_r),electric=-ne*b['phi_x'],
            pressure=chi*b['pressure_x'],Hall=-mu*b['omega_r_tau']*ne*vt,collision=mu*b['nu_tau']*ne*vx)
    tr=dict(inertia=mu*ne*(vx*vr_x+vr*vr_r),electric=-ne*b['phi_r'],pressure=chi*b['pressure_r'],Hall=mu*b['omega_x_tau']*ne*vt,collision=mu*b['nu_tau']*ne*vr)
    tt=dict(inertia=mu*ne*(vx*vt_x+vr*vt_r),Hall=mu*ne*(b['omega_r_tau']*vx-b['omega_x_tau']*vr),collision=mu*b['nu_tau']*ne*vt)
    return dict(electron_continuity=cont,electron_momentum_x=sum(tx.values()),electron_momentum_r=sum(tr.values()),
                electron_momentum_theta=sum(tt.values()),x_terms=tx,r_terms=tr,theta_terms=tt,velocities=(vx,vr,vt))

def anode_data(bg,d,p):
    rad,w=gl(32,d['r_min'],d['r_max'])
    x=torch.full((32,),d['x_min']);r=torch.tensor(rad)
    with torch.no_grad():f=bg(x,r)
    return dict(x=x,r=r,ni=f[1].detach(),vi=f[2].detach(),ne=f[5].detach(),
                area=torch.tensor(2*np.pi*rad*w*p['L_canal']**2))

def loss(net,x,r,b,p,anode,pressure):
    q=equations(net,x,r,b,p,pressure)
    raw={k:q[k].square().mean() for k in LABELS[:-1]}
    vex=net(anode['x'],anode['r'])[0]
    Id=p['e_phys']*p['n_ref']*p['V_ref']*((anode['ni']*anode['vi']-anode['ne']*vex)*anode['area']).sum()
    raw['anode_current']=((Id-5.4)/5.4).square()
    total=10*sum(raw.values())
    vals={k:float(v.detach()) for k,v in raw.items()}
    return total,dict(total=float(total.detach()),raw=vals,weighted={k:10*v for k,v in vals.items()},Id_anode_A=float(Id.detach()))

def append_loss(folder,epoch,values,opt,pre=None):
    row=dict(epoch=epoch,total=values['total'],lr=opt.param_groups[0]['lr'],Id_anode_A=values['Id_anode_A'],phase='post_update' if epoch else 'initial',preupdate_loss=pre,timestamp_utc=now())
    for k in LABELS:row.update({k+'_raw':values['raw'][k],k+'_weighted':values['weighted'][k],k+'_share_pct':100*values['weighted'][k]/values['total'] if values['total'] else 0})
    path=folder/'loss_terms.csv';new=not path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if new:w.writeheader()
        w.writerow(row);f.flush();os.fsync(f.fileno())

def snapshot(net,opt,sched,d,p,epoch,pressure,values,reason):
    return dict(model_state_dict=net.state_dict(),optimizer_state_dict=opt.state_dict(),scheduler_state_dict=sched.state_dict(),
                domain=d,physics=p,epochs_completed=epoch,pressure_enabled=pressure,pressure_coefficient=pressure_coefficient(p),
                background_checkpoint=str(BASE),background_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
                init_seed=7,collocation_seed=2601,torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),
                stop_reason=reason,saved_utc=now(),last_logged_loss=values,
                network='2 inputs; four hidden layers of 64 tanh; 3 electron velocity outputs; float64; Xavier uniform, zero biases')

def train(folder,pressure):
    folder=Path(folder);out=folder/'output';out.mkdir(exist_ok=True)
    assert not (folder/'loss_terms.csv').exists(),'Refuse to overwrite an existing run.'
    assert json.loads((HERE/'verification.json').read_text())['passed']
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_signal)
    bg,d,p,c=load_background();torch.manual_seed(7);np.random.seed(7);net=ElectronNet(d)
    torch.manual_seed(2601);np.random.seed(2601)
    x,r=a.utils.sample_domain_points(10000,d,'cpu')
    b=coefficients(bg,p,x,r);anode=anode_data(bg,d,p)
    assert all(not v.requires_grad for v in b.values())
    assert not any(par.requires_grad for par in bg.parameters())
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=300,min_lr=1e-6)
    state=dict(status='initializing',pid=os.getpid(),started_utc=now(),epochs_completed=0,requested_epochs=8000,
               pressure_enabled=pressure,pressure_coefficient=pressure_coefficient(p),threads=4,
               background_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest())
    write_json(folder/'progress.json',state);(folder/'pid.txt').write_text(str(os.getpid())+'\n')
    epoch=0;logged=-1;values=None;reason='completed 8000 epochs';start=time.perf_counter();last_pre=None
    try:
        value,values=loss(net,x,r,b,p,anode,pressure);del value
        append_loss(folder,0,values,opt);logged=0
        save_checkpoint(out/'checkpoint_epoch_0000.pt',snapshot(net,opt,sched,d,p,0,pressure,values,'initial'))
        state.update(status='training',loop_started_utc=now(),initial_loss=values['total']);write_json(folder/'progress.json',state)
        start=time.perf_counter()
        for step in range(1,8001):
            if STOP:reason=STOP;break
            opt.zero_grad();x.grad=None;r.grad=None
            value,pre=loss(net,x,r,b,p,anode,pressure);last_pre=pre['total']
            if not math_finite(last_pre):reason='nonfinite loss before optimizer update';del value;break
            value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();sched.step(last_pre)
            epoch=step;del value
            if epoch%100==0:
                value,values=loss(net,x,r,b,p,anode,pressure);del value
                append_loss(folder,epoch,values,opt,last_pre);logged=epoch
                print(f'epoch={epoch} total={values["total"]:.12g} Id_anode_A={values["Id_anode_A"]:.12g} lr={opt.param_groups[0]["lr"]:.6g}',flush=True)
                if not math_finite(values['total']):reason='nonfinite post-update loss';break
            if epoch%1000==0:
                save_checkpoint(out/f'checkpoint_epoch_{epoch:04d}.pt',snapshot(net,opt,sched,d,p,epoch,pressure,values,'block complete'))
            if epoch%100==0:
                elapsed=time.perf_counter()-start
                state.update(epochs_completed=epoch,elapsed_loop_seconds=elapsed,seconds_per_epoch=elapsed/epoch,
                             lr=opt.param_groups[0]['lr'],last_logged_loss=values,last_loss_epoch=logged)
                write_json(folder/'progress.json',state)
                from reporting import refresh
                refresh()
    except BaseException as exc:
        reason=f'operational failure: {type(exc).__name__}: {exc}';state['error']=reason;state['traceback']=traceback.format_exc();print(state['traceback'],flush=True)
    finally:
        # Save the current weights before any expensive or fallible diagnostics.
        save_checkpoint(out/'checkpoint.pt',snapshot(net,opt,sched,d,p,epoch,pressure,values,reason))
        save_checkpoint(out/f'checkpoint_stop_{epoch:04d}.pt',snapshot(net,opt,sched,d,p,epoch,pressure,values,reason))
        try:
            if logged!=epoch:
                value,values=loss(net,x,r,b,p,anode,pressure);del value
                append_loss(folder,epoch,values,opt,last_pre)
                save_checkpoint(out/'checkpoint.pt',snapshot(net,opt,sched,d,p,epoch,pressure,values,reason))
        except BaseException as exc:state['final_loss_error']=repr(exc)
        state.update(status='complete' if epoch==8000 else 'failed' if state.get('error') else 'stopped',
                     epochs_completed=epoch,stop_reason=reason,ended_utc=now(),training_wall_seconds=time.perf_counter()-start,
                     final_loss=values,background_unchanged=hashlib.sha256(BASE.read_bytes()).hexdigest()==state['background_sha256'])
        if state.get('loop_started_utc'):state['utc_elapsed_loop_seconds']=(datetime.datetime.fromisoformat(state['ended_utc'])-datetime.datetime.fromisoformat(state['loop_started_utc'])).total_seconds()
        write_json(folder/'progress.json',state)
    from reporting import refresh
    refresh()

def math_finite(v):return bool(np.isfinite(v))

def evaluate(folder):
    from wall_audit import converged
    folder=Path(folder);ck=torch.load(folder/'output/checkpoint.pt',map_location='cpu',weights_only=False)
    bg,d,p,_=load_background();net=ElectronNet(d);net.load_state_dict(ck['model_state_dict']);net.eval()
    pressure=ck['pressure_enabled'];x,r=a.grid(d);x.requires_grad_(True);r.requires_grad_(True)
    b=coefficients(bg,p,x,r);q=equations(net,x,r,b,p,pressure)
    residuals={k:dict(MSE=float(q[k].detach().square().mean()),RMS=float(q[k].detach().square().mean().sqrt())) for k in LABELS[:-1]}
    terms={k:float(v.detach().square().mean().sqrt()) for k,v in q['x_terms'].items()}
    other_terms={axis:{k:float(v.detach().square().mean().sqrt()) for k,v in q[axis+'_terms'].items()} for axis in ['r','theta']}
    ratios={axis:residuals['electron_momentum_'+axis]['RMS']/max(ts.values()) for axis,ts in dict(x=terms,**other_terms).items()}
    source_rms=float(b['Si'].square().mean().sqrt())
    vx,vr,vt=q['velocities'];mu=p['me']/p['mi'];chi=pressure_coefficient(p)
    transport=(1/mu)*b['nu_tau']/(b['nu_tau'].square()+b['omega_tau'].square())
    n=b['nu_tau']; ox=b['omega_x_tau']; omr=b['omega_r_tau'];z=torch.zeros_like(n)
    mat=torch.stack([n,z,-omr,z,n,ox,omr,-ox,n],dim=-1).reshape(-1,3,3)
    rhs=torch.stack([b['phi_x']-chi*b['pressure_x']/b['ne'],b['phi_r']-chi*b['pressure_r']/b['ne'],z],dim=-1)/mu
    drift_diffusion=torch.linalg.solve(mat,rhs.unsqueeze(-1)).squeeze(-1)[:,0]
    rhs0=torch.stack([b['phi_x'],b['phi_r'],z],dim=-1)/mu
    drift_no_pressure=torch.linalg.solve(mat,rhs0.unsqueeze(-1)).squeeze(-1)[:,0]
    signs=dict(Vex_negative_fraction=float((vx<0).double().mean()),Vex_positive_fraction=float((vx>0).double().mean()),
               exit_Vex_negative_fraction=float((vx.reshape(200,80)[-1]<0).double().mean()),
               sign_agreement_full_drift_diffusion=float((torch.sign(vx)==torch.sign(drift_diffusion)).double().mean()),
               sign_agreement_no_pressure_drift=float((torch.sign(vx)==torch.sign(drift_no_pressure)).double().mean()),
               sign_agreement_variant_prediction=float((torch.sign(vx)==torch.sign(drift_diffusion if pressure else drift_no_pressure)).double().mean()))
    ranges={name:dict(min=float(v.detach().min())*p['V_ref'],max=float(v.detach().max())*p['V_ref']) for name,v in zip(['Vex_m_s','Ver_m_s','Vetheta_m_s'],[vx,vr,vt])}
    del q,vx,vr,vt
    # Area-weighted cross-section RMS and signed integral at the same 40 stations.
    rad,wr=gl(128,d['r_min'],d['r_max']);stations=np.linspace(d['x_min'],d['x_max'],40)
    xx,rr=np.meshgrid(stations,rad,indexing='ij');xx=torch.tensor(xx.reshape(-1),requires_grad=True);rr=torch.tensor(rr.reshape(-1),requires_grad=True)
    bs=coefficients(bg,p,xx,rr);qs=equations(net,xx,rr,bs,p,pressure)
    area=torch.tensor(2*np.pi*rad*wr*p['L_canal']**2);profiles=[]
    for i,station in enumerate(stations):
        row=dict(x_hat=float(station))
        for key in LABELS[:-1]:
            v=qs[key].detach().reshape(40,128)[i]
            row[key+'_area_RMS']=float(((v.square()*area).sum()/area.sum()).sqrt())
        re=qs['electron_continuity'].detach().reshape(40,128)[i]
        row['electron_continuity_integral_A_per_m']=float((re*area).sum())*p['S_ref']*p['e_phys']
        profiles.append(row)
    del qs,bs
    composite=Composite(bg,net)
    wall=converged(composite,d,p)
    result=dict(pressure_enabled=pressure,pressure_coefficient=chi,epochs_completed=ck['epochs_completed'],
                residuals=residuals,x_momentum_term_RMS=terms,residual_over_largest_x_term_RMS=residuals['electron_momentum_x']['RMS']/max(terms.values()),
                other_momentum_term_RMS=other_terms,momentum_residual_over_largest_term_RMS=ratios,
                source_RMS=source_rms,continuity_RMS_over_source_RMS=residuals['electron_continuity']['RMS']/source_rms,
                signs=signs,velocity_ranges=ranges,axial_profiles=profiles,current_balance=wall,
                anode_current_error_fraction=(wall['final']['station_currents'][0]['Id_A']-5.4)/5.4)
    write_json(folder/'final_metrics.json',result)
    state=json.loads((folder/'progress.json').read_text());state['postprocessing']='complete';write_json(folder/'progress.json',state)
    from reporting import refresh
    refresh()
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('variant',choices=['A','B']);parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    # Isolated output: never the previous experiments/electron_frozen* folders.
    assert args.variant=='A','Only variant A is part of the anode_bc task.'
    folder=ION/'electron_frozen';folder.mkdir(exist_ok=True)
    if not args.evaluate_only:train(folder,args.variant=='A')
    evaluate(folder)
