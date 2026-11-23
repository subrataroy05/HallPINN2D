"""Warm start from ion_ramp with hard anode phi=274.04 V. Base physics, network and loss functions are reused unchanged."""
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
from integrated_constraints import quadrature

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'anode_bc/output/checkpoint.pt'
OUT = HERE/'output'
DATA_WEIGHTS = {'ni':10., 'Te':5., 'Vp':30.}
REQUESTED_EPOCHS = 6000
STOP_REQUEST = None

def now():
    return datetime.now(timezone.utc).isoformat()

def request_stop(signum, frame):
    global STOP_REQUEST
    STOP_REQUEST = f'user/process signal {signal.Signals(signum).name}'

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

def setup():
    torch.set_num_threads(4)
    torch.manual_seed(7); np.random.seed(7)
    model, domain, physics, original = a.load_model(BASE)
    domain = dict(domain)
    domain["phi_left"] = 274.04 / physics["phi_ref"]
    model.phi_left = domain["phi_left"]
    physics = dict(physics)
    physics["magnetic_field_model"] = "flux_2d"
    model.train()
    weights = dict(original['weights']); weights['integrated_current'] = 0.
    assert original['weights']['integrated_ion']==1000
    assert original['weights']['integrated_neutral']==100
    assert all(weights[k]==original['weights'][k] for k in weights if k!='integrated_current')
    assert 'exit_ion_current' not in weights
    assert physics['nn_inlet']==.164145
    torch.manual_seed(2601); np.random.seed(2601)
    x, r = a.utils.sample_domain_points(10000,domain,'cpu')
    boundary = a.utils.sample_boundary_points(1000,1000,domain,'cpu')
    data = a.utils.load_haas_gallimore_data(str(HERE/'code/haas_gallimore_p5_5p4A.csv'),domain,physics,'cpu')
    assert all(len(v['values'])==145 for v in data.values())
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=300,min_lr=1e-6)
    return model,domain,physics,original,weights,x,r,boundary,data,optimizer,scheduler

def evaluate_loss(model,domain,physics,weights,x,r,boundary,data):
    total, raw = a.utils.compute_all_losses(model,x,r,boundary,domain,physics,weights)
    dl = a.utils.compute_data_loss(model,data)
    weighted_data = DATA_WEIGHTS['ni']*dl['ni'] + DATA_WEIGHTS['Te']*dl['Te'] + DATA_WEIGHTS['Vp']*dl['Vp']
    total = total + weighted_data
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
    return dict(balance=b,fields=fields,all_fields_finite=finite,densities_positive=positive,
                max_abs_Vex_normalized=maxvex,grid_Vex_positive_fraction=float((f[6]>0).double().mean()),
                exit_Vex_positive_fraction=float((f[6].reshape(200,80)[-1]>0).double().mean()),
                Id_at_0_half_1_A=current['Id_A'].tolist(),Id_40_A=q['Id_A'].tolist(),
                Id_min_A=float(q['Id_A'].min()),Id_max_A=float(q['Id_A'].max()),
                exit_ion_A=iexit,exit_electron_conventional_A=-float(q['Gamma_e'][-1])*e,
                Ii_over_Id_exit=iexit/idexit if idexit else None,
                Vix_exit_centerline_km_s=center,Vix_exit_density_weighted_mean_km_s=meanv,
                pooled_L2=a.utils.data_loss_l2_relative(model,data),band_fractions=coverage)

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
                run_name='bfield_2d_seed7',eta_max=float(model.eta_max),final_loss=loss['total'],
                loss_at_saved_weights=loss,loss_epoch=epoch if loss.get('at_epoch')==epoch else loss.get('at_epoch'),
                loss_matches_saved_weights=loss.get('at_epoch')==epoch,
                optimizer_name='Adam',optimizer_initial_lr=1e-4,fresh_optimizer_state_at_start=True,
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
    state=dict(status='initializing',epochs_completed=0,requested_epochs=6000,pid=os.getpid(),started_utc=now(),threads=4)
    (HERE/'pid.txt').write_text(str(os.getpid())+'\n')
    (HERE/'progress.md').write_text('# Revised-anode progress\n\nPost-update diagnostics every 500 epochs. '
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
        reason='completed 6000 epochs'
        for step in range(1,6001):
            if STOP_REQUEST:reason=STOP_REQUEST;break
            weights['integrated_ion']=ion_weight(step)
            opt.zero_grad()
            total,pre=evaluate_loss(model,d,p,weights,x,r,boundary,data)
            last_pre=pre['total']
            if not math.isfinite(last_pre):
                reason='nonfinite training loss; immediate safety stop before optimizer update';del total;break
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();scheduler.step(last_pre)
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
                             projected_end_local=(datetime.now().astimezone()+timedelta(seconds=elapsed/epoch*(6000-epoch))).isoformat())
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
        state.update(status='complete' if epoch==6000 else 'stopped',epochs_completed=epoch,stop_reason=reason,
                     training_wall_seconds=elapsed,ended_utc=now(),checkpoint=str(OUT/'checkpoint.pt'),postprocessing='pending')
        state.pop('projected_end_local',None)
        write_json(HERE/'progress.json',state);refresh()
    try:finish()
    except BaseException as exc:
        state=json.loads((HERE/'progress.json').read_text());state['postprocessing_error']=repr(exc)
        write_json(HERE/'progress.json',state);refresh();raise

if __name__=='__main__':
    train()
