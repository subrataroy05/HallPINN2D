"""Roy 2 Nov: ion birth-location / collisionless exit-velocity distribution and exit flux check.
Assumptions: ions born at rest relative to neutrals (V_n) at (x',r) with rate S_i, move collisionlessly
and axially (no radial motion, no charge exchange, no wall loss), gaining e[phi(x',r)-phi(x_exit,r)]."""
import sys, json, csv
from pathlib import Path
F=Path(__file__).resolve().parent.parent/sys.argv[1]; sys.path.insert(0,str(F))
import numpy as np, torch, analyze as a
U=a.utils; torch.set_num_threads(2)
m,d,p,_=a.load_model(F/'output/checkpoint.pt')
e=1.602176634e-19; mi=2.1802e-25; L=p['L_canal']; Vref=p['V_ref']; nref=p['n_ref']; Vn=p['Vn']*Vref
nx,nr=801,96
xs=np.linspace(0,1,nx); zg,wg=np.polynomial.legendre.leggauss(nr)
rad=(d['r_min']+d['r_max'])/2+(d['r_max']-d['r_min'])*zg/2; wr=wg*(d['r_max']-d['r_min'])/2
X,R=np.meshgrid(xs,rad,indexing='ij')
with torch.no_grad():
    phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=m(torch.tensor(X.ravel()),torch.tensor(R.ravel()))
    Si=U.compute_Si(nn_,ne,Te,p)
sh=X.shape
phi=(phi*p['phi_ref']).numpy().reshape(sh); ni=(ni*nref).numpy().reshape(sh); vix=(Vix*Vref).numpy().reshape(sh)
Si=Si.numpy().reshape(sh)*p.get('S_ref',nref*Vref/L)
area=2*np.pi*R*L*L               # per unit normalized dr (multiply by wr)
dx=np.gradient(xs)*L
# exit flux and averages from the network
ex=-1
Ii=e*np.sum(ni[ex]*vix[ex]*area[ex]*wr)
vbar_n=np.sum(ni[ex]*vix[ex]*R[ex]*wr)/np.sum(ni[ex]*R[ex]*wr)
vbar_flux=np.sum(ni[ex]*vix[ex]**2*R[ex]*wr)/np.sum(ni[ex]*vix[ex]*R[ex]*wr)
imid=np.argmin(abs(rad-0.5*(d['r_min']+d['r_max'])))
# ionization totals
W=Si*area*wr[None,:]*dx[:,None]      # ions/s per cell
src=W.sum(); Isrc=e*src
gin=p['nn_inlet']*nref*p['Vn']*Vref*np.pi*(d['r_max']**2-d['r_min']**2)*L*L
# collisionless exit velocity of ions born in each cell (model potential)
def dist(phi_field):
    dphi=np.maximum(phi_field-phi_field[ex][None,:],0.0)
    v=np.sqrt(Vn**2+2*e*dphi/mi)
    w=W.ravel(); vv=v.ravel()
    fw=np.sum(w*vv)/np.sum(w); nw=np.sum(w)/np.sum(w/vv)
    hist,edges=np.histogram(vv/1e3,bins=np.arange(0,26,1.0),weights=w/np.sum(w))
    q=np.quantile(np.repeat(vv,1) if False else vv, [0.1,0.5,0.9]) # unweighted not used
    # weighted quantiles
    o=np.argsort(vv); cw=np.cumsum(w[o])/w.sum()
    wq=[float(vv[o][np.searchsorted(cw,f)]/1e3) for f in (0.1,0.5,0.9)]
    return dict(flux_mean_kms=fw/1e3,density_weighted_mean_kms=nw/1e3,w_quantiles_10_50_90_kms=wq,
                frac_below_8kms=float(np.sum(w[vv<8e3])/w.sum()),
                hist_kms=[[float(edges[k]),float(hist[k])] for k in range(len(hist))])
D_model=dist(phi)
# Haas Vp: per radius interpolation, nearest measured radius
rows=list(csv.DictReader(open(F/'code/haas_gallimore_p5_5p4A.csv')))
vp=[r for r in rows if r['quantity']=='Vp']
Rm=sorted(set(float(r['r_mm']) for r in vp))
phiH=np.zeros(sh)
for j,rr in enumerate(rad):
    Rmm=(rr-d['r_min'])*L*1000; Rn=min(Rm,key=lambda z:abs(z-Rmm))
    pts=sorted((float(r['x_mm']),float(r['value'])) for r in vp if float(r['r_mm'])==Rn)
    phiH[:,j]=np.interp(xs*L*1000,[a_ for a_,b in pts],[b for a_,b in pts])
D_data=dist(phiH)
xb=np.sum(W*X)/src; cum=np.cumsum(W.sum(1))/src
out=dict(run=sys.argv[1],
 network_exit=dict(I_i_A=Ii,I_i_over_5p4=Ii/5.4,expt_I_i_A=0.76*5.4,density_weighted_V_kms=vbar_n/1e3,flux_weighted_V_kms=vbar_flux/1e3,centerline_V_kms=float(vix[ex,imid]/1e3)),
 ionization=dict(source_A=Isrc,neutral_inflow_A=e*gin,utilization_source_over_inflow=src/gin,exit_ion_over_inflow=Ii/(e*gin),mean_birth_xhat=xb,
   frac_upstream={str(f):float(cum[int(f*(nx-1))]) for f in (0.3,0.5,0.7)}),
 ballistic_model_potential=D_model, ballistic_haas_potential=D_data, LIF_mean_kms=11.0)
O=Path(__file__).resolve().parent/('birth_'+sys.argv[1]+'.json'); json.dump(out,open(O,'w'),indent=1)
for k in ['network_exit','ionization']: print(k,json.dumps(out[k]))
for k in ['ballistic_model_potential','ballistic_haas_potential']:
    print(k,{kk:v for kk,v in out[k].items() if kk!='hist_kms'})
