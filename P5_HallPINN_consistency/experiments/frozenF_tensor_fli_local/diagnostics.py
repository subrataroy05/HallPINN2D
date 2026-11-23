"""Vector drift diagnostics, physical current splits and comparison reports."""
from pathlib import Path
import csv,json,sys,time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze as a
from field import B
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)

def save(name,value):
    (H/name).write_text(json.dumps(value,indent=2)+'\n')
def csvsave(name,rows):
    with (H/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):
    fmt=lambda v:f'{v:.10g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in rows)+'\n'
def figsave(fig,name):
    fig.tight_layout();fig.savefig(H/name,dpi=150,bbox_inches='tight');plt.close(fig)

def local(m,p,x,r,force_2d=None):
    x=x.detach().clone().requires_grad_(True);r=r.detach().clone().requires_grad_(True)
    f=m(x,r);phi,ni,vi,_,nn,ne,_,_,te,_,_=f
    grad=lambda y,z:torch.autograd.grad(y.sum(),z,retain_graph=True)[0]
    gx=grad(phi,x);gr=grad(phi,r);px=grad(ne*te,x)/ne;pr=grad(ne*te,r)/ne
    use2d=p.get('magnetic_field_model')=='flux_2d' if force_2d is None else force_2d
    bx,br=B(x,r) if use2d else (torch.zeros_like(x),a.utils.compute_Br(x,p))
    mag=torch.hypot(bx,br);bhx=bx/mag;bhr=br/mag
    omega=p['e_over_me']*mag;ncl=a.utils.compute_collision_frequencies(ne,nn,te,p)['nu_total'];nperp=ncl+p['alpha_B']*omega
    npar=ncl if p.get('transport_tensor',False) else nperp;nu=npar;z=torch.zeros_like(ncl)
    n=nperp*p['tau_ref'];dn=(npar-nperp)*p['tau_ref'];ox=bx*p['e_over_me']*p['tau_ref'];orr=br*p['e_over_me']*p['tau_ref']
    mat=torch.stack([n+dn*bhx*bhx,dn*bhx*bhr,-orr,dn*bhx*bhr,n+dn*bhr*bhr,ox,orr,-ox,n],dim=-1).reshape(-1,3,3)
    rhs=torch.stack([gx-px,gr-pr,z],dim=-1)*(p['mi']/p['me'])
    v=torch.linalg.solve(mat,rhs.unsqueeze(-1)).squeeze(-1)
    vp=(v[:,0]*bhx+v[:,1]*bhr);vparx=vp*bhx
    # Requested voltage-gradient convention: E_parallel + grad_parallel(pe)/(e ne).
    fpar=(bhx*(-gx+px)+bhr*(-gr+pr))*p['phi_ref']/p['L_canal']
    expected=-(p['e_phys']/p['me'])*fpar/nu/p['V_ref']
    # Absolute and relative check, including points where parallel force nearly cancels.
    assert torch.allclose(vp,expected,rtol=2e-10,atol=1e-7)
    return {k:v.detach().numpy() for k,v in dict(ni=ni,ne=ne,vi=vi,vx=v[:,0],vr=v[:,1],vtheta=v[:,2],vparx=vparx,vperpx=v[:,0]-vparx,Fpar_V_m=fpar,phi_x_V_m=gx*p['phi_ref']/p['L_canal'],omega_over_nu=omega/nu,bx=bhx,br=bhr).items()}

def sections(m,d,p,stations,nr=2048,force_2d=None,rlo=None,rhi=None):
    lo=d['r_min'] if rlo is None else rlo;hi=d['r_max'] if rhi is None else rhi
    z,w=np.polynomial.legendre.leggauss(nr);rad=(lo+hi)/2+(hi-lo)*z/2;w=w*(hi-lo)/2
    area=2*np.pi*rad*w*p['L_canal']**2;scale=p['e_phys']*p['n_ref']*p['V_ref'];rows=[]
    for station in stations:
        q=local(m,p,torch.full((nr,),float(station)),torch.tensor(rad),force_2d)
        ion=float(np.sum(q['ni']*q['vi']*area)*scale)
        par=-float(np.sum(q['ne']*q['vparx']*area)*scale);perp=-float(np.sum(q['ne']*q['vperpx']*area)*scale)
        rows.append(dict(x_hat=float(station),ion_A=ion,electron_parallel_conventional_A=par,electron_perpendicular_Hall_conventional_A=perp,Id_DD_A=ion+par+perp))
    return rows

def a6():
    start=time.perf_counter();m,d,p,_=a.load_model(H.parent/'anode_bc/output/checkpoint.pt')
    rows=sections(m,d,p,np.linspace(0,1,40),force_2d=True);csvsave('A6_current_split.csv',rows)
    x,r=a.grid(d);q=local(m,p,x,r,True);F=q['Fpar_V_m'];rms=float(np.sqrt(np.mean(F*F)))
    coords=np.linspace(0,1,201);profiles=[]
    for rad in [.25,2.5,12.5,22.5,24.75]:
        q=local(m,p,torch.tensor(coords),torch.full((len(coords),),d['r_min']+rad/1000/p['L_canal']),True)
        profiles.extend(dict(x_hat=float(x),R_mm=rad,Fpar_V_m=float(f)) for x,f in zip(coords,q['Fpar_V_m']))
    csvsave('A6_parallel_force_profiles.csv',profiles)
    bands=[]
    for lo,hi in zip(np.arange(0,25,5),np.arange(5,30,5)):
        v=sections(m,d,p,[1],512,True,d['r_min']+lo/1000/p['L_canal'],d['r_min']+hi/1000/p['L_canal'])[0]
        bands.append(dict(R_lo_mm=int(lo),R_hi_mm=int(hi),**{k:z for k,z in v.items() if k!='x_hat'},fraction_exit_Id=v['Id_DD_A']/rows[-1]['Id_DD_A']))
    csvsave('A6_exit_radial_bands.csv',bands)
    rad=np.linspace(d['r_min'],d['r_max'],1001);q=local(m,p,torch.ones(1001),torch.tensor(rad),True)
    scale=p['e_phys']*p['n_ref']*p['V_ref']*2*np.pi*rad*p['L_canal']*.001
    ion=q['ni']*q['vi']*scale;par=-q['ne']*q['vparx']*scale;perp=-q['ne']*q['vperpx']*scale
    radial=[dict(R_mm=float((r-d['r_min'])*p['L_canal']*1000),ion_A_mm=float(i),parallel_A_mm=float(a_),perpendicular_Hall_A_mm=float(b),total_A_mm=float(i+a_+b)) for r,i,a_,b in zip(rad,ion,par,perp)]
    csvsave('A6_exit_radial_distribution.csv',radial)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for key,label in [('electron_parallel_conventional_A','Electron parallel'),('electron_perpendicular_Hall_conventional_A','Electron perpendicular + Hall'),('ion_A','Ion'),('Id_DD_A','Total')]:axes[0].plot([v['x_hat'] for v in rows],[v[key] for v in rows],label=label)
    axes[0].set(xlabel='x_hat',ylabel='Current contribution (A)',yscale='symlog');axes[0].legend(fontsize=8)
    for radius in [.25,12.5,24.75]:
        rs=[v for v in profiles if v['R_mm']==radius];axes[1].plot(coords,[v['Fpar_V_m'] for v in rs],label=f'R={radius:g} mm')
    axes[1].set(xlabel='x_hat',ylabel='Parallel voltage-gradient imbalance (V/m)');axes[1].legend(fontsize=8)
    for key in ['parallel_A_mm','perpendicular_Hall_A_mm','total_A_mm']:axes[2].plot([v['R_mm'] for v in radial],[v[key] for v in radial],label=key)
    axes[2].set(xlabel='Radius from inner wall (mm)',ylabel='Exit current density per radial mm (A/mm)');axes[2].legend(fontsize=8);figsave(fig,'A6_split.png')
    summary=dict(Fpar_grid_RMS_V_m=rms,current_split=rows,exit_bands=bands,exit_band_sum_error_A=sum(v['Id_DD_A'] for v in bands)-rows[-1]['Id_DD_A'],wall_seconds=time.perf_counter()-start)
    save('A6_results.json',summary)
    lines=['## A6 — parallel/perpendicular split\n',
      '**User-authorized gate override:** the 3.09 G radial fit RMS is accepted on the user’s stated probe-position justification: ±1 mm with gradients up to approximately 13 G/mm near the inner wall (user citation: Haas 2001, p.106). This justification is recorded as supplied; the fit, coefficients and curl weights have not been changed. The original numerical gate failure remains documented above; it no longer blocks Part B.\n',
      'All A6 fields are from the final frozen anode_bc checkpoint with the unchanged main 2D magnetic fit. Define v_parallel=(v·b)b and v_perpendicular+Hall=v−v_parallel. The table reports conventional electron current contributions −e∫ne v_parallel,x dA and −e∫ne v_perpendicular,x dA; the ion current is shown separately. Thus I_DD=I_ion+I_e,parallel+I_e,perpendicular+Hall. It would double-count the ion current to add two separate total discharge currents.\n',
      'The sign convention for the requested force diagnostic is **F_parallel=b·[−grad(phi)+grad(ne Te)/(e ne)]**, in V/m, with Te converted from eV. It is the voltage-gradient imbalance E_parallel+pressure-gradient/(e ne), not a force in newtons. Therefore v_parallel_scalar=−e F_parallel/(me nu). This equality is checked against the 3×3 solution. Hall velocity has no projection along B.\n',
      f'Uniform inclusive 200×80 grid RMS of F_parallel: **{rms:.10g} V/m**. Near-wall profiles are evaluated 0.25 mm inside each wall; mid-radius is 12.5 mm. The CSV additionally includes 2.5 and 22.5 mm. GL2048 radial quadrature is used for the 40 station fluxes.\n',table(list(rows[0]),[list(v.values()) for v in rows]),
      '### Exit radial contributions\n','Five 5-mm annular bands span the full channel. All contributions are signed. The radial distribution CSV contains 1001 samples in A/mm.\n',table(list(bands[0]),[list(v.values()) for v in bands]),
      f'Band-integral sum minus full exit integral: {summary["exit_band_sum_error_A"]:.6g} A.\n',
      '[Parallel force profiles](A6_parallel_force_profiles.csv) · [Exit distribution](A6_exit_radial_distribution.csv) · [Split plots](A6_split.png).\n',f'A6 wall time: {summary["wall_seconds"]:.6g} s.\n']
    (H/'A6.md').write_text('\n'.join(lines))
    from reporting import refresh
    refresh();print(json.dumps({k:v for k,v in summary.items() if k not in ['current_split','exit_bands']},indent=2))

def final_compare():
    import experiment as e
    names=['ion_ramp','anode_bc','bfield_2d'];runs={};models={};stations=np.linspace(0,1,40)
    for name in names:
        path=H.parent/name/'output/checkpoint.pt';m,d,p,c=a.load_model(path);models[name]=(m,d,p)
        data=a.utils.load_haas_gallimore_data(str(H/'code/haas_gallimore_p5_5p4A.csv'),d,p,'cpu')
        met=a.measure(path);diag=e.diagnostic(m,d,p,data)
        with torch.no_grad():
            ds=data['Vp'];mask=ds['coords'][:,0]>.3;co=ds['coords'][mask];pred=m(co[:,0],co[:,1])[0];tar=ds['values'][mask];vp=float(torch.linalg.vector_norm(pred-tar)/torch.linalg.vector_norm(tar))
        torch.manual_seed(2601);np.random.seed(2601);x,r=a.utils.sample_domain_points(10000,d,'cpu');boundary=a.utils.sample_boundary_points(1000,1000,d,'cpu')
        t,loss=e.evaluate_loss(m,d,p,c['weights'],x,r,boundary,data);del t
        runs[name]=dict(metrics=met,diagnostic=diag,Vp_L2_xhat_gt_0p3=vp,final_loss=loss,DD=sections(m,d,p,stations))
    save('comparison.json',runs)
    from reporting import flatten
    flat={n:flatten(runs[n]['diagnostic']) for n in names};lines=['## Part B — final plasma comparison\n',table(['Metric']+names,[[k]+[flat[n].get(k) for n in names] for k in flat[names[0]]]),
      table(['Metric']+names,[['Vp L2 x_hat >0.3']+[runs[n]['Vp_L2_xhat_gt_0p3'] for n in names]])]
    for stat in ['MSE','RMS']:lines+=[table(['Unweighted residual '+stat]+names,[[k]+[runs[n]['metrics']['residuals'][k][stat] for n in names] for k in a.KEYS])]
    lines += [table(['x_hat']+[n+' I_DD A' for n in names],[[float(x)]+[runs[n]['DD'][i]['Id_DD_A'] for n in names] for i,x in enumerate(stations)]),table(['x_hat']+[n+' network Id A' for n in names],[[float(x)]+[runs[n]['diagnostic']['Id_40_A'][i] for n in names] for i,x in enumerate(stations)])]
    for name in names:lines += [f'### {name}: final loss terms\n',table(['Term','Raw','Weighted','Share %'],[[k,runs[name]['final_loss']['raw'][k],v,runs[name]['final_loss']['shares_percent'][k]] for k,v in runs[name]['final_loss']['weighted'].items()])]
    fig,axes=plt.subplots(2,2,figsize=(12,8));profile=[]
    for name,(m,d,p) in models.items():
        xp=np.linspace(0,.35,141);q=local(m,p,torch.tensor(xp),torch.full((len(xp),),(d['r_min']+d['r_max'])/2))
        axes[0,0].plot(xp,q['phi_x_V_m'],label=name);axes[0,1].plot(xp,q['omega_over_nu'],label=name);axes[1,0].plot(xp,q['vx']*p['V_ref'],label=name);axes[1,1].plot(stations,[v['Id_DD_A'] for v in runs[name]['DD']],label=name)
        profile.extend(dict(run=name,x_hat=float(x),dphi_dx_V_m=float(g),omega_over_nu=float(o),Vex_DD_m_s=float(v)*p['V_ref']) for x,g,o,v in zip(xp,q['phi_x_V_m'],q['omega_over_nu'],q['vx']))
    for ax,y in zip(axes.flat,['dphi/dx V/m','omega/nu','Vex DD m/s','I_DD A']):ax.set(xlabel='x_hat',ylabel=y);ax.legend(fontsize=8)
    axes[1,1].set_yscale('symlog');figsave(fig,'final_DD_profiles.png');csvsave('final_DD_profiles.csv',profile)
    fig,axs=plt.subplots(1,5,figsize=(18,4),sharey=True);figmid,axmid=plt.subplots(figsize=(8,4))
    for j,radmm in enumerate([2.5,7.5,12.5,17.5,22.5]):
        for name,(m,d,p) in models.items():
            xp=torch.linspace(0,1,200);rad=d['r_min']+radmm/1000/p['L_canal']
            with torch.no_grad():yp=m(xp,torch.full_like(xp,rad))[0].numpy()*p['phi_ref']
            axs[j].plot(xp.numpy(),yp,label=name)
            if j==2:axmid.plot(xp.numpy(),yp,label=name)
        ds=data['Vp'];mask=torch.isclose(ds['coords'][:,1],torch.tensor(rad));xs=ds['coords'][mask,0].numpy();ys=ds['values'][mask].numpy()*p['phi_ref']
        for ax in ([axs[j],axmid] if j==2 else [axs[j]]):ax.scatter(xs,ys,s=10,c='k',label='Haas data');ax.fill_between(xs,ys-3,ys+6,color='grey',alpha=.25);ax.set(xlabel='x_hat',ylabel='Vp (V)')
        axs[j].set_title(f'R={radmm:g} mm')
    axs[0].legend(fontsize=7);axmid.legend();figsave(fig,'Vp_five_radii_comparison.png');figsave(figmid,'Vp_midradius_comparison.png')
    lines+=['[Full DD profiles](final_DD_profiles.png) · [Vp five radii](Vp_five_radii_comparison.png) · [Mid-radius Vp with −3/+6 V band](Vp_midradius_comparison.png).\n','Four inherited figures: [centerline](figures/bfield_2d_seed7_centerline.png), [ni](figures/ch5_hg_ni_5radii.png), [Te](figures/ch5_hg_Te_5radii.png), [Vp](figures/ch5_hg_Vp_5radii.png). [Independent full Gauss audit](final_partA.json).\n']
    (H/'final_comparison.md').write_text('\n'.join(lines))
    from reporting import refresh
    refresh()

def frozen_compare():
    folders=[H.parent/'electron_frozen',H.parent/'anode_bc/electron_frozen',H/'electron_frozen'];names=['ion_ramp background A','anode_bc background A','2D-field background A'];metrics=[json.loads((f/'final_metrics.json').read_text()) for f in folders]
    rows=[]
    for k in ['Id_min_A','Id_max_A','Id_mean_A']:rows.append([k]+[v['current_balance']['final'][k] for v in metrics])
    rows += [['Id span A']+[v['current_balance']['final']['Id_max_A']-v['current_balance']['final']['Id_min_A'] for v in metrics],['Continuity RMS/source RMS']+[v['continuity_RMS_over_source_RMS'] for v in metrics]]
    for k in ['x','r','theta']:rows.append([k+' momentum residual/largest term']+[v['momentum_residual_over_largest_term_RMS'][k] for v in metrics])
    for k in metrics[0]['signs']:rows.append([k]+[v['signs'][k] for v in metrics])
    for k in metrics[0]['residuals']:rows.append([k+' RMS']+[v['residuals'][k]['RMS'] for v in metrics])
    lines=['## Pressure-inclusive frozen-electron A comparison\n','All three comparisons use the original 8000-epoch A budgets on their respective frozen backgrounds; no extended/restarted A checkpoint is substituted. The new electron network has fresh Xavier initialization seed 7, four 64-unit tanh layers, float64, fixed seed-2601 collocation, 4 CPU threads, 8000 Adam steps at 1e-3, usual scheduler, clipping 1 and five weights of 10. The new frozen engine uses the full signed 2D magnetic Lorentz terms and |B| in the Bohm rate. Pressure enters x and r with coefficient 1; anode current alone is targeted at 5.4 A.\n',table(['Metric']+names,rows)]
    fig,ax=plt.subplots(figsize=(8,4))
    for name,v in zip(names,metrics):cs=v['current_balance']['final']['station_currents'];ax.plot([q['x_hat'] for q in cs],[q['Id_A'] for q in cs],label=name)
    ax.axhline(5.4,c='k',ls=':');ax.set(xlabel='x_hat',ylabel='Network I_d (A)');ax.legend();figsave(fig,'frozen_current_comparison.png')
    lines += ['[Frozen-electron current comparison](frozen_current_comparison.png) · [Full new frozen metrics and interval balance](electron_frozen/final_metrics.json).\n']
    for name,folder in zip(names,folders):st=json.loads((folder/'progress.json').read_text());lines += [f'{name}: {st["status"]}, {st["epochs_completed"]}/8000 steps, training wall {st.get("training_wall_seconds")} s.\n']
    new=metrics[-1];w=new['current_balance']['final'];phys=json.loads((H/'comparison.json').read_text())['bfield_2d'];di=phys['diagnostic']
    lines += ['## Conclusion\n',f'The 2D-field plasma run gives ion imbalance {100*di["balance"]["ion_imbalance_over_source"]:.6g}% and neutral imbalance {100*di["balance"]["neutral_defect_over_inflow"]:.6g}%; exit Vix {di["Vix_exit_centerline_km_s"]:.6g} km/s. Its network discharge current spans {di["Id_min_A"]:.6g}–{di["Id_max_A"]:.6g} A. The separate frozen-electron fit spans {w["Id_min_A"]:.6g}–{w["Id_max_A"]:.6g} A, with continuity RMS/source RMS {new["continuity_RMS_over_source_RMS"]:.6g} and momentum residual/largest-term ratios {new["momentum_residual_over_largest_term_RMS"]}. These numbers show the remaining conservation and force-balance errors; completing optimization does not make them vanish. The field-fit gate was explicitly overridden, and the substantial upstream extrapolation uncertainty remains.\n']
    (H/'frozen_comparison.md').write_text('\n'.join(lines))
    from reporting import refresh
    refresh()

if __name__=='__main__':{'a6':a6,'final':final_compare,'frozen':frozen_compare}[sys.argv[1]]()
