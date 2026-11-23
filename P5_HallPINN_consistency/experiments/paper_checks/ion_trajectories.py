"""Roy 3 Nov: collisional ion trajectory calculation on a trained run's fields (no training).
Ions are born at the model ionization rate S_i(x,r) with the neutral velocity, move in the model 2D electric field
(axial and radial), undergo resonant Xe+-Xe charge exchange with the model neutral density (Miller et al. 2002 cross
section; a CEX event replaces the ion by a slow ion at the neutral velocity), and are lost at the walls or the anode.
Variants: collisionless axial-only (as in the earlier estimate), collisionless 2D, and 2D with charge exchange."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, torch
RUN=Path(sys.argv[1]).resolve(); CK=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else RUN/'output/checkpoint.pt'
sys.path.insert(0,str(RUN)); sys.path.insert(0,str(RUN/'code'))
import analyze as a
U=a.utils; torch.set_num_threads(4)
m,d,p,_=a.load_model(CK); m.eval()
L=p['L_canal']; e=1.602176634e-19; mi=p['mi']; Vref=p['V_ref']; phiref=p['phi_ref']
nx,nr=401,121
xs=np.linspace(0,1,nx); rs=np.linspace(d['r_min'],d['r_max'],nr)
XX,RR=np.meshgrid(xs,rs,indexing='ij')
x=torch.tensor(XX.ravel()).requires_grad_(True); r=torch.tensor(RR.ravel()).requires_grad_(True)
f=m(x,r); phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=f
g=lambda y,z: torch.autograd.grad(y.sum(),z,retain_graph=True)[0]
Ex=(-g(phi,x)*phiref/L).detach().numpy().reshape(nx,nr); Er=(-g(phi,r)*phiref/L).detach().numpy().reshape(nx,nr)
Si=U.compute_Si(nn_,ne,Te,p).detach().numpy().reshape(nx,nr)
NN=(nn_*p['n_ref']).detach().numpy().reshape(nx,nr); PHI=(phi*phiref).detach().numpy().reshape(nx,nr)
Vn=float(p['Vn'])*Vref
def interp(F,X_,R_):
    fx=np.clip(X_,0,1)*(nx-1); fr=np.clip((R_-d['r_min'])/(d['r_max']-d['r_min']),0,1)*(nr-1)
    i=np.clip(fx.astype(int),0,nx-2); j=np.clip(fr.astype(int),0,nr-2); tx=fx-i; tr=fr-j
    return (1-tx)*(1-tr)*F[i,j]+tx*(1-tr)*F[i+1,j]+(1-tx)*tr*F[i,j+1]+tx*tr*F[i+1,j+1]
def sigma_cex(vrel):  # Miller et al. 2002, Xe+ + Xe, energy in eV, cross-section in m^2
    E=np.clip(0.5*mi*vrel**2/e,1.0,None); return (87.3-13.6*np.log10(E))*1e-20
rng=np.random.default_rng(3); N=30000
w=Si*RR; w=w/w.sum(); idx=rng.choice(w.size,N,p=w.ravel())
X0=XX.ravel()[idx]+rng.uniform(-.5,.5,N)/(nx-1); R0=RR.ravel()[idx]+rng.uniform(-.5,.5,N)*(rs[1]-rs[0])
def run(mode):
    X=X0.copy();R=R0.copy();vx=np.full(N,Vn);vr=np.zeros(N);alive=np.ones(N,bool);fate=np.zeros(N,int)  # 1 exit, 2 wall, 3 anode
    vexit=np.full(N,np.nan);ncex=np.zeros(N,int);dt=2e-9
    for _ in range(40000):
        if not alive.any():break
        k=alive; ax=e/mi*interp(Ex,X[k],R[k]); ar=0. if mode=='axial' else e/mi*interp(Er,X[k],R[k])
        vx[k]+=ax*dt; vr[k]+=ar*dt; X[k]+=vx[k]*dt/L; R[k]+=vr[k]*dt/L
        if mode=='cex':
            vrel=np.hypot(vx[k]-Vn,vr[k]); P=interp(NN,X[k],R[k])*sigma_cex(vrel)*vrel*dt
            hit=rng.random(k.sum())<P; ids=np.where(k)[0][hit]; vx[ids]=Vn; vr[ids]=0.; ncex[ids]+=1
        ex=alive&(X>=1); vexit[ex]=vx[ex]; fate[ex]=1; alive&=~ex
        wl=alive&((R<=d['r_min'])|(R>=d['r_max'])); fate[wl]=2; alive&=~wl
        an=alive&(X<=0); fate[an]=3; alive&=~an
    v=vexit[fate==1]
    q=lambda a_: float(np.percentile(v,a_))/1e3 if len(v) else None
    return dict(frac_exit=float(np.mean(fate==1)),frac_wall=float(np.mean(fate==2)),frac_anode=float(np.mean(fate==3)),frac_unfinished=float(np.mean(fate==0)),
                flux_mean_km_s=float(v.mean())/1e3,density_weighted_mean_km_s=float(len(v)/np.sum(1/v))/1e3,p10=q(10),p50=q(50),p90=q(90),
                frac_below_8km_s=float(np.mean(v<8e3)),mean_cex_per_exit_ion=float(ncex[fate==1].mean()) if mode=='cex' else 0.,
                hist_km_s=np.histogram(v/1e3,bins=np.arange(0,31,1))[0].tolist())
out=dict(run=str(RUN.name),checkpoint=str(CK.name),N=N,Vn_m_s=Vn,birth_mean_x_hat=float(np.average(XX.ravel()[idx])),
         nn_exit_mean=float(NN[-1].mean()),phi_max_V=float(PHI.max()),phi_exit_mean_V=float(PHI[-1].mean()),
         axial_collisionless=run('axial'),collisionless_2d=run('2d'),cex_2d=run('cex'))
json.dump(out,open(Path(__file__).resolve().parent/f'ion_traj_{RUN.name}.json','w'),indent=1)
for k in ('axial_collisionless','collisionless_2d','cex_2d'):
    o=out[k]; print(k,{a_:(round(b,3) if isinstance(b,float) else b) for a_,b in o.items() if a_!='hist_km_s'})
