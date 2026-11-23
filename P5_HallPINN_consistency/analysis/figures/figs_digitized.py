"""Figures for the digitized-data sensitivity subsection:
 (1) digitized_vs_reconstruction.pdf : AIAA-2000-3422 Figs 5-9 (vector extraction) vs analytic reconstruction
 (2) fieldline_mismatch_digitized.pdf : field-line mismatch M for both datasets, total-uncertainty bars."""
import csv, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.labelsize':12,'legend.fontsize':12,
                     'xtick.labelsize':10,'ytick.labelsize':10})
def load(p):
    D={}
    for r in csv.DictReader(open(p)):
        D.setdefault((r['quantity'],float(r['r_mm'])),[]).append((float(r['x_mm']),float(r['value'])))
    return {k:np.array(sorted(v)).T for k,v in D.items()}
dig=load('../haas_digitized.csv')        # raw extracted points
rec=load('haas_gallimore_p5_5p4A.csv')     # analytic reconstruction (training targets)
R=[2.5,7.5,12.5,17.5,22.5]
Q=[('Vp',r'$V_p$ (V)',1),('Te',r'$T_e$ (eV)',1),('ni',r'$n_i$ ($10^{18}$ m$^{-3}$)',1e-18)]
fig,ax=plt.subplots(3,5,figsize=(15,8.2),sharex=True)
for i,(q,lab,sc) in enumerate(Q):
    for j,r in enumerate(R):
        a=ax[i,j]; a.axvspan(10,38,color='0.92',zorder=0)
        x,v=dig[(q,r)]; a.plot(x,v*sc,'k.-',ms=3,lw=0.8,label='digitized (AIAA-2000-3422)')
        x,v=rec[(q,r)]; a.plot(x,v*sc,'r-',lw=1.4,label='analytic reconstruction')
        if i==0: a.set_title(f'$r-r_{{in}}$ = {r} mm',fontsize=12)
        if j==0: a.set_ylabel(lab)
        if i==2: a.set_xlabel('$x$ (mm)')
        a.set_xlim(8,82)
h,l=ax[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=2,fontsize=13,frameon=False,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig('digitized_vs_reconstruction.pdf'); fig.savefig('digitized_vs_reconstruction.png',dpi=200)

s=json.load(open('sensitivity.json')); t=json.load(open('total_mc.json'))
s0=json.load(open('sensitivity.json')); t0=json.load(open('total_mc.json'))
x=np.array(s['seeds_mm'])
M=np.array(s['radial_interpolation']['linear']); sd=np.array(t['per_radius']['std'])
M0=np.array(s0['radial_interpolation']['linear']); sd0=np.array(t0['per_radius']['std'])
fig,a=plt.subplots(figsize=(7.5,4.2))
a.axhline(0,color='0.5',lw=0.8)
a.errorbar(x-0.25,M0,yerr=2*sd0,fmt='s--',color='0.6',ms=5,capsize=3,lw=1.4,elinewidth=1,label='analytic reconstruction (training targets)')
a.errorbar(x+0.25,M,yerr=2*sd,fmt='o-',color='#1f4e79',ms=5,capsize=3,lw=1.6,label='digitized AIAA-2000-3422 profiles')
a.axvline(29,color='#b22222',ls=':',lw=1.2); a.text(29.3,-42,'$B$ peak',color='#b22222',fontsize=11)
a.set_xlabel('field-line seed $x$ at mid-radius (mm)'); a.set_ylabel(r'$M=\Delta V_p-\int dp_e/en_e$ (V)')
a.set_ylim(-50,95); a.legend(loc='upper left',fontsize=12,frameon=False)
fig.tight_layout(); fig.savefig('fieldline_mismatch_digitized.pdf'); fig.savefig('fieldline_mismatch_digitized.png',dpi=200)
