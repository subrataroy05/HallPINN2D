import csv, numpy as np, matplotlib
from pathlib import Path
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent; B=H.parent/'bfield_2d'; O=H.parents[1]/'thesis_appendix_figs'
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':0.3})
rd=lambda f: list(csv.DictReader(open(B/f)))
c=lambda r,k: np.array([float(x[k]) for x in r])
s=rd('A6_current_split.csv'); p=rd('A6_parallel_force_profiles.csv'); d=rd('A6_exit_radial_distribution.csv')
fig,ax=plt.subplots(1,3,figsize=(13,3.9))
x=c(s,'x_hat')
ax[0].semilogy(x,np.abs(c(s,'electron_parallel_conventional_A')),label='Electron, field-aligned')
ax[0].semilogy(x,np.abs(c(s,'electron_perpendicular_Hall_conventional_A')),label='Electron, cross-field and Hall')
ax[0].semilogy(x,np.abs(c(s,'ion_A')),label='Ion')
ax[0].set(xlabel=r'$\hat{x}$',ylabel='Current contribution magnitude (A)'); ax[0].legend(fontsize=8)
for R,lab in [(0.25,'0.25 mm from inner wall'),(12.5,'mid-radius'),(24.75,'0.25 mm from outer wall')]:
    q=[r for r in p if float(r['R_mm'])==R]; ax[1].plot(c(q,'x_hat'),c(q,'Fpar_V_m'),label=lab)
ax[1].set(xlabel=r'$\hat{x}$',ylabel=r'$R_\parallel$ (V/m)'); ax[1].legend(fontsize=8)
ax[2].plot(c(d,'R_mm'),c(d,'parallel_A_mm'),label='Field-aligned')
ax[2].plot(c(d,'R_mm'),c(d,'perpendicular_Hall_A_mm'),label='Cross-field and Hall')
ax[2].set(xlabel='Distance from inner wall (mm)',ylabel='Exit current per mm (A/mm)'); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(O/'appF_parallel_split.png',dpi=200); print('ok')
