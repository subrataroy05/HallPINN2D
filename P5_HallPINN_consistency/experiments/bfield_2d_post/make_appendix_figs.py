import csv, shutil
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent; O=H.parents[1]/'thesis_appendix_figs'; O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':0.3})
def rd(run,f): return list(csv.DictReader(open(H/('roy_'+run)/f)))
def col(rows,k): return np.array([float(r[k]) for r in rows])
# 1. Vp error vs x
fig,ax=plt.subplots(figsize=(6.2,3.6))
for run,lab,st in [('bfield_2d','Bounded potential ($A=25$)','o-'),('bfield_2d_phifix','Enlarged potential ($2A$)','s-'),('stageC_current','Final configuration (stage C)','^-')]:
    v=rd(run,'vp_error.csv'); ax.plot(col(v,'x_hat'),col(v,'Vp_err_mean_V'),st,ms=4,label=lab)
ax.axhspan(-3,6,color='grey',alpha=0.15,label='Measurement band $-3/+6$ V')
ax.set(xlabel=r'$\hat{x}$',ylabel=r'$V_p$ model $-$ data (V)'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(O/'appF_vp_error.png',dpi=200); plt.close(fig)
# 2. ion momentum budget
fig,axs=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
for ax,run,t in [(axs[0],'bfield_2d_phifix','Enlarged potential, density data weighted'),(axs[1],'phifix_no_ni_data','Enlarged potential, density data removed')]:
    s=rd(run,'stations.csv'); x=col(s,'x_hat')
    ax.plot(x,col(s,'ion_electric'),label='Electric'); ax.plot(x,col(s,'ion_convective'),label='Convective')
    ax.plot(x,col(s,'ion_in_exchange')+col(s,'ion_ionization'),label='Ion-neutral + ionization'); ax.plot(x,col(s,'ion_residual'),'k--',label='Residual')
    ax.set(title=t,xlabel=r'$\hat{x}$'); ax.title.set_fontsize(9)
axs[0].set_ylabel('Cross-section integral (normalized)'); axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(O/'appF_ion_momentum_budget.png',dpi=200); plt.close(fig)
# 3. implied vs network current
fig,ax=plt.subplots(figsize=(6.2,3.8))
for run,lab,c in [('stageA_uw_pressure','Stage A','C0'),('stageB3_par_continue','Stage B','C1'),('stageC_current','Stage C','C2')]:
    v=rd(run,'current_compare.csv'); x=col(v,'x_hat')
    ax.semilogy(x,np.abs(col(v,'I_DD_total_A')),'-',color=c,label=lab+r': $|I_{DD}|$ implied')
    ax.semilogy(x,np.maximum(np.abs(col(v,'I_network_A')),1e-2),':',color=c,label=lab+': network')
ax.axhline(5.4,color='k',lw=0.8); ax.text(0.01,6.2,'5.4 A',fontsize=8)
ax.set(xlabel=r'$\hat{x}$',ylabel='Current magnitude (A)'); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(O/'appF_current_compare.png',dpi=200); plt.close(fig)
# 4. R_par vs x
fig,ax=plt.subplots(figsize=(6.2,3.6))
for run,lab in [('bfield_2d','Measured field, before constraint'),('stageB2_par_ramp','Stage B (end of ramp)'),('stageB3_par_continue','Stage B (converged)'),('stageC_current','Stage C')]:
    s=rd(run,'stations.csv'); ax.plot(col(s,'x_hat'),col(s,'Rpar_rms_V_m'),label=lab)
ax.set(xlabel=r'$\hat{x}$',ylabel=r'RMS $R_\parallel$ over cross-section (V/m)'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(O/'appF_rpar.png',dpi=200); plt.close(fig)
# copy existing
for src,dst in [('bfield_2d/bfield_fit.png','appF_bfield_fit.png'),('bfield_2d/bfield_maps.png','appF_bfield_maps.png'),('bfield_2d/A6_split.png','appF_parallel_split.png')]:
    shutil.copy(H.parent/src,O/dst)
print(sorted(p.name for p in O.iterdir()))
