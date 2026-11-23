"""One 15000-epoch CPU run; observations do not alter the optimizer loop."""
import ast,hashlib,json,os,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'output';OUT.mkdir(exist_ok=True)
for key in ('EPOCHS_OVERRIDE','LBFGS_STEPS_OVERRIDE','OPTIMIZER_OVERRIDE','SMOOTH_NI_OVERRIDE','CENTERLINE_DATA_R_MM'):
    os.environ.pop(key,None)
os.environ.update(SEED_OVERRIDE='7',CKPT_IN_OVERRIDE='',OUTPUT_OVERRIDE=str(OUT),
                  CKPT_OUT_OVERRIDE=str(OUT/'checkpoint.pt'),SUMMARY_OVERRIDE=str(OUT/'validation.md'),
                  MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/mplconfig',PYTHONPYCACHEPREFIX='/private/tmp/roy_pycache')
os.chdir(HERE/'code')
sys.path.insert(0,str(HERE))
from analyze import driver,utils,diagnostic,measure,generate_figures,full_partA
from report import refresh
import torch
torch.set_num_threads(4)
driver.RUN_NAME='roy_current_seed7'
assert driver.INIT_SEED==7 and driver.SEED==2601 and driver.EPOCHS_TO_RUN==15000 and driver.LBFGS_STEPS==0
assert driver.NEUTRAL_INLET_FROM_MASS_FLOW and not torch.cuda.is_available()
assert json.loads((HERE/'verification.json').read_text())['passed']
assert not (OUT/'checkpoint.pt').exists()
manifest=json.loads((HERE/'base_source_sha256.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
checkpoint_paths=[ROOT/'het_pinn_vfinal/experiments/bohm_momentum_ab/run_d/output/run_d_alpha1over200_vetheta2000_seed7_checkpoint.pt',HERE.parent/'nu_in_local/output/checkpoint.pt']
checkpoint_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_paths}
tree=ast.parse((HERE/'code/mainScript.py').read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='k')
after=main.body[main.body.index(loop)+1]
state=dict(status='initializing',epochs_completed=0,init_seed=7,collocation_seed=2601,threads=torch.get_num_threads(),
           device='cpu',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),base_checkpoint_sha256=checkpoint_hashes)
(HERE/'pid.txt').write_text(str(os.getpid())+'\n')
progress_md=HERE/'progress.md'
assert not progress_md.exists(),'Do not overwrite an existing run.'
progress_md.write_text('# Roy current experiment: current-weight progress\n\n'
 'Seed 7 / collocation 2601; CPU, 4 Torch threads. Diagnostics after each 1000 optimizer steps. '
 'Global imbalances include anode ion influx (unlike the literal requested ion training penalty). '
 'Ion denominator = volume source; neutral denominator = inlet neutral flux. '
 'Physical fluxes use the training quadrature; final Part A independently doubles the quadrature.\n\n'
 '| Epoch | Exit Vix (km/s) | Id(0) A | Id(0.5) A | Id(1) A | Exit ion A | Grid Vex>0 fraction | ni pooled L2 | Ion global imbalance | Neutral global imbalance |\n'
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
logged=set();start=None

def save():
    temp=HERE/'progress.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(HERE/'progress.json');refresh()

def report_weights(frame,epoch):
    if not epoch or epoch%1000 or epoch in logged:return
    env=frame.f_locals;m=env['model'];was_training=m.training
    values=diagnostic(m,env['domain'],env['physics'],env['exp_data'])
    assert m.training==was_training
    print(f'[current weights epoch {epoch:5d}] '+json.dumps(values),flush=True)
    row=[epoch,values['Vix_exit_km_s'],*values['Id_A'],values['ion_exit_A'],values['grid_Vex_positive_fraction'],
         values['ni_pooled_L2'],values['ion_imbalance_over_source'],values['neutral_defect_over_inflow']]
    with progress_md.open('a') as f:f.write('| '+' | '.join(f'{v:.12g}' for v in row)+' |\n')
    logged.add(epoch);state['latest_diagnostic']=dict(epoch=epoch,**values)
    with (HERE/'progress_diagnostics.jsonl').open('a') as f:f.write(json.dumps(state['latest_diagnostic'])+'\n')

def trace_main(frame,event,arg):
    global start
    if event=='line':
        if frame.f_lineno==loop.lineno:
            if start is None:
                start=time.perf_counter();state.update(status='training',loop_started_utc=datetime.now(timezone.utc).isoformat());save()
            done=frame.f_locals.get('final_epoch',0)
            if done and done%25==0:
                report_weights(frame,done)
                elapsed=time.perf_counter()-start
                state.update(epochs_completed=done,elapsed_loop_seconds=elapsed,seconds_per_epoch_so_far=elapsed/done,
                             projected_end_local=(datetime.now().astimezone()+timedelta(seconds=elapsed/done*(15000-done))).isoformat())
                save()
        elif frame.f_lineno==after.lineno and start is not None:
            done=frame.f_locals['final_epoch'];report_weights(frame,done)
            state.update(status='postprocessing',epochs_completed=done,training_wall_seconds=time.perf_counter()-start,
                         loop_ended_utc=datetime.now(timezone.utc).isoformat(),stop_reason=frame.f_locals['stop_reason'])
            state['seconds_per_epoch']=state['training_wall_seconds']/done
            save();sys.settrace(None);return None
    return trace_main

def trace_calls(frame,event,arg):
    return trace_main if event=='call' and frame.f_code is driver.main.__code__ else None

def postprocess():
    result=measure(OUT/'checkpoint.pt')
    (HERE/'final_metrics.json').write_text(json.dumps(result,indent=2)+'\n');save()
    audit=full_partA(OUT/'checkpoint.pt')
    (HERE/'final_partA.json').write_text(json.dumps(audit,indent=2)+'\n');save()
    figures=generate_figures(OUT/'checkpoint.pt')
    (HERE/'figures.json').write_text(json.dumps(figures,indent=2)+'\n');save()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in checkpoint_hashes.items())
    state.update(status='complete; figures awaiting visual inspection' if state['epochs_completed']==15000 else 'stopped early; final audit saved',
                 bases_unchanged=True)
    save()

save()
try:
    sys.settrace(trace_calls);driver.main();sys.settrace(None)
    state['status']='evaluating final checkpoint';save();postprocess()
except BaseException as exc:
    sys.settrace(None);state.update(status='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
