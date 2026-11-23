"""Constraint-compatibility map (Roy 3 Nov): for each tested formulation, which constraints are met."""
import json, csv, statistics as st
from pathlib import Path
E=Path(__file__).resolve().parents[1]; P=E/'bfield_2d_post'
runs=[('Measured magnetic field','bfield_2d','iso',None),('Enlarged potential','bfield_2d_phifix','iso',None),
      ('Enlarged potential, no density data','phifix_no_ni_data','iso',None),('Stage A: uncertainty + pressure','stageA_uw_pressure','iso',None),
      ('Stage B: local field-aligned (E_ref)','stageB3_par_continue','iso',None),('Stage C: + 5.4 A current','stageC_current','iso',None),
      ('Electron-only, local field-aligned (frozenC)','frozenC_electron','iso',None),('Electron-only, mobility-weighted (frozenD)','frozenD_mobility','iso',None),
      ('Electron-only, tensor + FLI + weak local (frozenF)','frozenF_tensor_fli_local','tensor',None),('Fully coupled, tensor + FLI + weak local (G)','coupledG_tensor_fli','tensor',5500)]
def last_diag(folder,epoch=None):
    f=E/folder/'progress_diagnostics.jsonl'
    if not f.exists(): return None
    L=[json.loads(l) for l in open(f)]
    if epoch is not None: L=[d for d in L if d['epoch']==epoch] or L
    return L[-1]
def idd(folder,d):
    if d and 'I_DD_40_A' in d: I=d['I_DD_40_A']
    else:
        f=P/('roy_'+folder)/'current_compare.csv'
        if not f.exists(): return None
        I=[float(r['I_DD_total_A']) for r in csv.DictReader(open(f))]
    x=[i/39 for i in range(40)]
    return dict(anode_max=max(abs(v) for v,q in zip(I,x) if q<.3),mid_median=st.median(abs(v) for v,q in zip(I,x) if .45<=q<=.85),exit_max=max(abs(v) for v,q in zip(I,x) if q>.9))
def energy(folder,epoch):
    for name in ([f'energy_{folder}_checkpoint_epoch_{epoch}.json'] if epoch else [])+[f'energy_{folder}.json',f'energy_{folder}_final.json']:
        f=P/name
        if f.exists():
            m=json.load(open(f))['r_frac_0.5']; return dict(KE_eV=m['dKE_eV'],electric_eV=m['electric_work_eV'],residual_eV=m['residual_work_eV'],res_over_el=m['ion_res_rms_over_electric_rms'])
    return None
rows=[]
for label,folder,closure,ep in runs:
    d=last_diag(folder,ep); b=d['band_fractions'] if d else {}; bal=d['balance'] if d else {}
    rp=d.get('Rpar_rms_grid_V_m') if d else None
    if rp is None and (P/('roy_'+folder)/'summary.json').exists(): rp=json.load(open(P/('roy_'+folder)/'summary.json'))['Rpar_rms_overall_V_m']
    I=idd(folder,d); en=energy(folder,ep)
    row=dict(run=label,folder=folder,closure=closure,epoch=d['epoch'] if d else None,
        band_ni=b.get('ni'),band_Te=b.get('Te'),band_Vp=b.get('Vp'),ion_imb=bal.get('ion_imbalance_over_source'),neut_imb=bal.get('neutral_defect_over_inflow'),
        Id_min=d.get('Id_min_A') if d else None,Id_max=d.get('Id_max_A') if d else None,Rpar_V_m=rp,
        IDD_anode=I and I['anode_max'],IDD_mid=I and I['mid_median'],IDD_exit=I and I['exit_max'],
        Vix_exit=d.get('Vix_exit_density_weighted_mean_km_s') if d else None,Ii_over_Id=d.get('Ii_over_Id_exit') if d else None,
        KE_eV=en and en['KE_eV'],residual_work_eV=en and en['residual_eV'])
    ok=lambda v,c: None if v is None else bool(c(v))
    row['C_data']=ok(row['band_Vp'],lambda v: v>=.9 and row['band_Te']>=.8 and row['band_ni']>=.8)
    row['C_data_noNi']=ok(row['band_Vp'],lambda v: v>=.9 and row['band_Te']>=.8)
    row['C_particles']=ok(row['ion_imb'],lambda v: abs(v)<=.05 and abs(row['neut_imb'])<=.05)
    row['C_current']=ok(row['Id_min'],lambda v: v>=4.86 and row['Id_max']<=5.94)
    row['C_electron_mom']=ok(row['IDD_anode'],lambda v: max(v,row['IDD_mid'],row['IDD_exit'])<=10.8)
    row['C_ion_energy']=ok(row['KE_eV'],lambda v: abs(row['residual_work_eV'])<=0.2*v)
    rows.append(row)
json.dump(rows,open(Path(__file__).parent/'compat_map.json','w'),indent=1)
with open(Path(__file__).parent/'compat_map.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fmt=lambda v,s='%.2f': '—' if v is None else (s%v if not isinstance(v,bool) else ('YES' if v else 'no'))
for r in rows:
    print(r['run'][:46].ljust(46),'bands',fmt(r['band_ni']),fmt(r['band_Te']),fmt(r['band_Vp']),'imb',fmt(r['ion_imb'],'%.3f'),fmt(r['neut_imb'],'%.3f'),'Id',fmt(r['Id_min']),fmt(r['Id_max']),
          'Rpar',fmt(r['Rpar_V_m'],'%.0f'),'IDD',fmt(r['IDD_anode'],'%.0f'),fmt(r['IDD_mid'],'%.0f'),fmt(r['IDD_exit'],'%.0f'),'Vix',fmt(r['Vix_exit'],'%.1f'),'KE/res',fmt(r['KE_eV'],'%.0f'),fmt(r['residual_work_eV'],'%.0f'),
          '|',' '.join(f"{k[2:]}={fmt(r[k])}" for k in r if k.startswith('C_')))
