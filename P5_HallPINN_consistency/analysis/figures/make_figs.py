import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e',
  'axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,'grid.color':'#e6e5e1','grid.linewidth':0.6,'savefig.dpi':300})
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
INK2='#52514e'
# ---------------- Fig 1: constraint-compatibility chart ----------------
cols=['Ion density\n(±50%)','$T_e$ / $V_p$\n(±20%, −3/+6 V)','Particle\nconservation','Discharge\ncurrent 5.4 A','Electron\nmomentum','Ion energy\nconsistency']
P,F,M,N='pass','fail','partial','na'
rows=[('Reference (Gaussian $B$)',[(P,'98%'),(F,'91/66%'),(F,'−40%'),(P,'closure'),(N,'n/e'),(F,'under')]),
 ('Measured $B$',[(F,'48%'),(F,'94/52%'),(P,'−0.3/−0.9%'),(F,'1.3–4.2 A'),(F,'4·10$^4$ A'),(F,'under')]),
 ('Enlarged potential',[(F,'46%'),(P,'82/97%'),(P,'−1.1/−1.8%'),(F,'−2.3–3.3 A'),(F,'4·10$^4$ A'),(F,'under')]),
 ('No density data',[(F,'31%'),(P,'86/97%'),(P,'−1.3/+3.9%'),(F,'−1.8–3.3 A'),(F,'2·10$^4$ A'),(P,'20%')]),
 ('A: uncertainty + $p_e$',[(F,'38%'),(P,'90/96%'),(P,'−0.7/+3.6%'),(F,'−1.5–2.9 A'),(F,'2·10$^4$ A'),(P,'5%')]),
 ('B: local field-aligned',[(F,'25%'),(P,'87/95%'),(P,'−2.2/+0.7%'),(F,'−0.3–2.5 A'),(F,'3·10$^3$ A'),(P,'16%')]),
 ('C: + current',[(F,'10%'),(P,'88/97%'),(P,'−2.7/−2.3%'),(P,'5.1–5.4 A'),(F,'2·10$^3$ A'),(F,'66% over')]),
 ('e$^-$-only, local',[(F,'10%'),(P,'88/97%'),(P,'−3.4/−2.8%'),(P,'5.3–5.4 A'),(F,'2·10$^3$ A'),(N,'frozen')]),
 ('e$^-$-only, mobility',[(F,'10%'),(F,'88/81%'),(F,'−6.0/−4.6%'),(F,'4.6–5.3 A'),(F,'113 A'),(N,'frozen')]),
 ('e$^-$-only, tensor + FLI',[(F,'10%'),(P,'84/97%'),(F,'−5.9/−4.5%'),(P,'5.1–5.5 A'),(F,'1.3·10$^3$ A'),(N,'frozen')]),
 ('Coupled, tensor + FLI',[(F,'6%'),(P,'91/97%'),(P,'−3.6/−1.5%'),(M,'5.3–5.9 A*'),(F,'1.4·10$^3$ A'),(F,'83% over')])]
col={P:'#0ca30c',F:'#d03b3b',M:'#fab219',N:'#c9c7c0'}; sym={P:'✓',F:'✗',M:'~',N:'–'}
fig,ax=plt.subplots(figsize=(7.4,4.9)); ax.grid(False); CW=1.18
for i,(name,cells) in enumerate(rows):
    for j,(s,t) in enumerate(cells):
        ax.add_patch(Rectangle((j*CW+0.04,-i-0.92),CW-0.08,0.84,fc=col[s],ec='white',alpha=0.22 if s!=N else 0.5,lw=0))
        ax.text(j*CW+0.1,-i-0.5,sym[s],ha='left',va='center',fontsize=9,color=col[s] if s!=N else INK2,fontweight='bold',fontfamily='DejaVu Sans')
        ax.text(j*CW+0.36,-i-0.5,t,ha='left',va='center',fontsize=6.8,color='#0b0b0b')
    ax.text(-0.08,-i-0.5,name,ha='right',va='center',fontsize=8)
for j,c in enumerate(cols): ax.text(j*CW+CW/2,0.12,c,ha='center',va='bottom',fontsize=7.4,color='#0b0b0b')
for y in (-1,-7): ax.axhline(y,color=INK2,lw=0.8)
ax.set_xlim(-3.0,6*CW);ax.set_ylim(-11.05,1.25);ax.axis('off')
fig.text(0.01,0.01,'Electron momentum: largest |implied current| near the anode or exit (target ≤ 2×5.4 A).\nIon energy: residual work as a share of the ion kinetic-energy gain at mid-radius (target ≤ 20%); "under"/"over": ions under- or over-accelerated.\n*oscillating under the current penalty. n/e: not evaluated (Gaussian field, no pressure term).',fontsize=6.3,color=INK2)
plt.tight_layout(rect=(0,0.08,1,1)); fig.savefig('figs/compat_map.pdf'); fig.savefig('figs/compat_map.png'); plt.close()
# ---------------- Fig 2: field-line mismatch and required parallel current ----------------
seeds=np.arange(12,37,2.0)
main=[28.2,35.8,38.3,25.1,13.4,5.4,2.0,1.4,-0.5,-3.2,-8.4,-11.1,-5.4]
var={'Reduced curl penalty':[28.2,35.6,38.3,25.1,13.3,5.4,2.0,1.4,-0.5,-5.1,-8.5,-11.4,-5.7],
     'Radial near-anode field':[30.3,40.8,31.8,21.3,11.6,4.4,1.6,0.6,-0.9,-4.5,-9.4,-11.9,-6.4],
     'Constant near-anode field':[28.3,35.3,40.0,26.1,14.1,5.9,2.3,1.7,0.8,-2.7,-7.8,-10.6,-5.3]}
te08=[23.95,30.41,32.81,22.68,13.77,7.99,5.77,5.81,4.81,2.85,-1.32,-4.0,-0.2]; te12=[32.51,41.27,43.8,27.53,12.95,2.9,-1.85,-3.1,-5.78,-9.29,-15.38,-18.25,-10.69]
import json
_ic=json.load(open('../../experiments/paper_checks/independent_checks.json'))
mc_pr=_ic['monte_carlo']['per_radius_2sig']['std_V']
assert [round(v,1) for v in _ic['M0_V']]==[round(v,1) for v in [28.23,35.84,38.3,25.11,13.36,5.44,1.96,1.35,-0.49,-3.22,-8.35,-11.13,-5.44]]
Jr=[9.85e6,9.98e6,9.69e6,7.62e6,4.36e6,1.77e6,6.11e5,4.09e5,1.48e5,1.05e6,2.71e6,3.23e6,1.3e6]
Jr1=[4.44e7,4.51e7,4.46e7,3.55e7,2.06e7,8.44e6,2.92e6,1.94e6,6.86e5,4.59e6,1.11e7,1.22e7,4.51e6]
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.0))
a.fill_between(seeds,te08,te12,color=C[0],alpha=0.13,lw=0,label='$T_e$ scaled by 0.8–1.2')
a.axhline(0,color=INK2,lw=0.8); a.axvspan(11,20,color='#eeeeea',zorder=0)
for (k,v),c,mk in zip(var.items(),[C[1],C[2],C[6]],['s','^','D']): a.plot(seeds,v,mk,ms=4,color=c,mfc='none',mew=1,label=k)
a.errorbar(seeds,main,yerr=2*np.array(mc_pr),fmt='none',ecolor=C[0],elinewidth=1,capsize=2.5)
a.plot(seeds,main,'o-',color=C[0],ms=4.5,lw=1.6,label='Reconstructed field')
a.set(xlabel='Field line (axial position at mid-radius, mm)',ylabel=r'$\Delta V_p-\int dp_e/en_e$ (V)'); a.legend(fontsize=6.3,frameon=False,loc='upper right')
a.text(11.4,-17,'upstream',fontsize=7,color=INK2); a.text(0.02,0.96,'(a)',transform=a.transAxes,fontweight='bold',va='top')
b.semilogy(seeds,np.abs(Jr),'o-',color=C[0],ms=4.5,lw=1.6,label=r'$n_n=5.5\times10^{18}$ m$^{-3}$')
b.semilogy(seeds,np.abs(Jr1),'s--',color=C[1],ms=4,lw=1.2,mfc='none',label=r'$n_n=1\times10^{18}$ m$^{-3}$')
b.axhline(474,color=INK2,lw=1,ls=':'); b.text(21,620,'discharge current density 474 A m$^{-2}$',fontsize=7,color=INK2)
b.axvspan(11,20,color='#eeeeea',zorder=0)
b.set(xlabel='Field line (axial position at mid-radius, mm)',ylabel=r'required $|J_\parallel|$ (A m$^{-2}$)',ylim=(1e2,1e9)); b.legend(fontsize=6.8,frameon=False,loc='upper right')
b.text(0.02,0.96,'(b)',transform=b.transAxes,fontweight='bold',va='top')
plt.tight_layout(); fig.savefig('figs/fieldline_mismatch.pdf'); fig.savefig('figs/fieldline_mismatch.png'); plt.close()
# ---------------- Fig 3: term-by-term parallel balance ----------------
locs=['Anode\n$\\hat x$=0.05, 46 G','B peak\n$\\hat x$=0.72, 102 G','Exit\n$\\hat x$=0.97, 91 G']
terms={'Electric':[708,-1680,-767],'Pressure':[-924,1660,694],'Drag (classical + Bohm)':[0.34,0.157,0.068],'Inertia':[-0.027,-31.1,-20.8],'Residual':[-216,-55.1,-93.3]}
fig,ax=plt.subplots(figsize=(6.0,2.9)); w=0.16
for k,(name,v) in enumerate(terms.items()):
    x=np.arange(3)+(k-2)*w; vals=np.abs(v)
    ax.bar(x,vals,w*0.88,color=C[[0,1,2,3,7][k]] if name!='Residual' else '#0b0b0b',label=name)
    for xi,vi,si in zip(x,vals,v): ax.text(xi,vi*1.25,'+' if si>0 else '−',ha='center',fontsize=7,color=INK2)
ax.set_yscale('log'); ax.set_ylim(1e-3,2e4); ax.set_xticks(range(3)); ax.set_xticklabels(locs,fontsize=8)
ax.set_ylabel('|term| (V m$^{-1}$, per electron)'); ax.legend(fontsize=7,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.16))
plt.tight_layout(); fig.savefig('figs/parallel_terms.pdf'); fig.savefig('figs/parallel_terms.png'); plt.close()
# ---------------- Fig 4: ion energy budget and trajectories ----------------
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw=dict(width_ratios=[1,1.25]))
lab=['Stage A','Stage C','Coupled\ntensor']; el=[158,158,158]; res=[8,213,355]; oth=[4,-50,-87]; dK=[170,322,426]
x=np.arange(3); wd=0.22
a.bar(x-wd,el,wd*0.9,color=C[0],label='Electric work'); a.bar(x,res,wd*0.9,color=C[7],label='Residual work'); a.bar(x+wd,oth,wd*0.9,color=C[3],label='Other physical terms')
a.plot(x,dK,'D',color='#0b0b0b',ms=5,label=r'Kinetic energy gain $\Delta K$'); a.axhline(0,color=INK2,lw=0.8)
a.set_xticks(x);a.set_xticklabels(lab);a.set_ylabel('eV per ion (mid-radius)');a.set_ylim(-130,680);a.legend(fontsize=6.3,frameon=False,loc='upper left',ncol=1)
a.text(0.97,0.97,'(a)',transform=a.transAxes,fontweight='bold',va='top',ha='right')
rowsT=[('Stage C, axial collisionless',12.03,15.04,15.39,14.27),('Stage C, 2D collisionless',12.05,15.04,15.39,14.27),('Stage C, 2D + charge exchange',11.78,15.01,15.39,14.18),
       ('Coupled, axial collisionless',11.75,15.05,15.36,14.17),('Coupled, 2D collisionless',11.74,15.04,15.36,14.17),('Coupled, 2D + charge exchange',11.37,15.00,15.36,14.05)]
for i,(n,p10,p50,p90,mn) in enumerate(rowsT):
    y=-i; b.plot([p10,p90],[y,y],color=C[0] if i<3 else C[2],lw=3,solid_capstyle='round'); b.plot(p50,y,'|',color='#0b0b0b',ms=9,mew=1.5); b.plot(mn,y,'o',color='white',mec='#0b0b0b',ms=4.5)
    b.text(8.6,y,n,ha='right',va='center',fontsize=7)
b.axvline(11,color=C[6],lw=1.2,ls='--'); b.text(11.1,0.55,'LIF 11',fontsize=7,color=C[6])
b.plot(21.1,-1,'v',color=C[7],ms=6); b.plot(24.4,-4,'v',color=C[7],ms=6); b.text(21.3,-0.75,'fluid C',fontsize=7,color=C[7]); b.text(22.4,-3.65,'fluid coupled',fontsize=7,color=C[7])
b.set_xlim(8.7,26); b.set_ylim(-5.6,1.1); b.set_yticks([]); b.set_xlabel('Exit ion velocity (km s$^{-1}$)'); b.grid(axis='y',visible=False)
b.text(0.97,0.96,'(b)',transform=b.transAxes,fontweight='bold',va='top',ha='right')
plt.tight_layout(); fig.savefig('figs/ion_energy_traj.pdf'); fig.savefig('figs/ion_energy_traj.png'); plt.close()
print('ok')
