"""Total uncertainty of M: measurement errors (three error models, bands as 2 sigma) + axial probe position
(+-1.5 mm as 2 sigma, per radius) + radial interpolation drawn among linear/spline/quadratic and lin/log density."""
import numpy as np, json
import sensitivity as S
rng=np.random.default_rng(23); N=2000; nR=len(S.R)
def perturbed(kind):
    sn,sT,sV=0.25,0.10,2.25
    nX=len(S.XF)
    if kind=='independent':
        en=rng.normal(0,sn,(nX,nR));eT=rng.normal(0,sT,(nX,nR));eV=rng.normal(0,sV,(nX,nR))
    elif kind=='per_radius':
        en=np.repeat(rng.normal(0,sn,(1,nR)),nX,0);eT=np.repeat(rng.normal(0,sT,(1,nR)),nX,0);eV=np.repeat(rng.normal(0,sV,(1,nR)),nX,0)
    else:
        rho=(S.R-S.R.mean())/(S.R.max()-S.R.mean())
        en=rng.normal(0,sn)*rho+rng.normal(0,sn/2,(nX,nR));eT=rng.normal(0,sT)*rho+rng.normal(0,sT/2,(nX,nR));eV=rng.normal(0,sV)*rho+rng.normal(0,sV/2,(nX,nR))
    return {'ni':S.GF['ni']*np.exp(en),'Te':np.clip(S.GF['Te']*(1+eT),0.5,None),'Vp':S.GF['Vp']+eV}
M0=np.array(S.out["radial_interpolation"]["linear"])
res={}
for kind in ('independent','per_radius','radial_tilt'):
    A=[]
    for _ in range(N):
        G=perturbed(kind); m=['linear','spline'][rng.integers(2)]
        A.append(np.array([S.M_line(P,method=m,logn=bool(rng.integers(2)),dx=rng.normal(0,0.75,nR),G=G) for P in S.lines]))
    A=np.array(A); sd=(np.percentile(A,84.13,0)-np.percentile(A,15.87,0))/2
    res[kind]=dict(std=sd.round(2).tolist(),z=(M0/sd).round(2).tolist(),P_sign=[float(np.mean(np.sign(A[:,k])!=np.sign(M0[k]))) for k in range(len(M0))])
    print(kind,"std",sd.round(1),"z",(M0/sd).round(2),"P",[round(v,3) for v in res[kind]["P_sign"]])
json.dump(res,open('total_mc.json','w'),indent=1)
