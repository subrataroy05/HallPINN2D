"""Roy 3 Nov 20:57, items 2 and 3, PINN-free.
(2) Upstream field-line mismatch M = dVp - int dp_e/n_e versus propagated measurement uncertainty, with
    independent, per-probe-radius systematic, and smooth radial-tilt error models.
(3) Independent solution of the parallel electron momentum relation along fixed field lines from the
    reconstructed profiles: e n dphi/ds = dp_e/ds + m_e n nu_par V_par, giving the uniform parallel current
    density J_par that would reconcile the measured potential at both ends of each line."""
import csv, json
from pathlib import Path
import numpy as np
import field as FM
H=Path(__file__).resolve().parent; model=json.load(open(H/'field_main.json'))
e=1.602176634e-19; me=9.1093837e-31
rows=list(csv.DictReader(open(H/'haas_gallimore_p5_5p4A.csv')))
X=np.array(sorted({float(r['x_mm']) for r in rows if float(r['x_mm'])<=38})); R=np.array(sorted({float(r['r_mm']) for r in rows}))
G={q:np.zeros((len(X),len(R))) for q in ('ni','Te','Vp')}
for r in rows:
    if float(r['x_mm'])<=38: G[r['quantity']][list(X).index(float(r['x_mm'])),list(R).index(float(r['r_mm']))]=float(r['value'])
def bdir(x,r):
    m=FM.evaluate(x/1000,0.06+r/1000,model); n=np.hypot(m['Bx'],m['Br']); return m['Bx']/n,m['Br']/n
def trace(seeds,ds=0.05):
    out={}
    for sgn in (-1,1):
        x=seeds.copy();r=np.full(len(seeds),12.5);act=np.ones(len(x),bool);P=[[(a,b)] for a,b in zip(x,r)]
        for _ in range(4000):
            if not act.any():break
            f=lambda x,r: tuple(sgn*v for v in bdir(x,r))
            k1=f(x,r);k2=f(x+ds/2*k1[0],r+ds/2*k1[1]);k3=f(x+ds/2*k2[0],r+ds/2*k2[1]);k4=f(x+ds*k3[0],r+ds*k3[1])
            xn=x+ds/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);rn=r+ds/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            act=act&(X[0]<=xn)&(xn<=X[-1])&(R[0]<=rn)&(rn<=R[-1])
            for k in np.where(act)[0]:P[k].append((xn[k],rn[k]))
            x=np.where(act,xn,x);r=np.where(act,rn,r)
        out[sgn]=P
    return [np.array(out[-1][k][::-1]+out[1][k][1:]) for k in range(len(seeds))]
seeds=np.arange(12,37,2.0); lines=trace(seeds)
def weights(P):
    x=np.clip(P[:,0],X[0],X[-1]);r=np.clip(P[:,1],R[0],R[-1])
    i=np.clip(np.searchsorted(X,x)-1,0,len(X)-2);j=np.clip(np.searchsorted(R,r)-1,0,len(R)-2)
    tx=(x-X[i])/(X[i+1]-X[i]);tr=(r-R[j])/(R[j+1]-R[j]); return i,j,tx,tr
W=[weights(P) for P in lines]
def ev(g,w):
    i,j,tx,tr=w; return (1-tx)*(1-tr)*g[...,i,j]+tx*(1-tr)*g[...,i+1,j]+(1-tx)*tr*g[...,i,j+1]+tx*tr*g[...,i+1,j+1]
def M_of(n,T,V):  # arrays (..., nX, nR) -> list of (...,) per line
    out=[]
    for w in W:
        nn=ev(n,w);TT=ev(T,w);VV=ev(V,w);p=nn*TT
        out.append((VV[...,-1]-VV[...,0])-np.sum(np.diff(p,axis=-1)/(0.5*(nn[...,1:]+nn[...,:-1])),axis=-1))
    return np.array(out)
M0=M_of(G['ni'],G['Te'],G['Vp'])
# ---------- (2) Monte Carlo uncertainty propagation ----------
rng=np.random.default_rng(11); N=4000; nX,nR=len(X),len(R)
def draw(kind,scale):
    """scale: stated bands treated as 2-sigma ('2sig') or 1-sigma ('1sig')."""
    k=1.0 if scale=='1sig' else 0.5
    sn,sT,sV=0.5*k,0.2*k,4.5*k; vshift=1.5   # Vp band -3/+6 V: centre +1.5 V, half-width 4.5 V
    if kind=='independent':
        en=rng.normal(0,sn,(N,nX,nR));eT=rng.normal(0,sT,(N,nX,nR));eV=rng.normal(0,sV,(N,nX,nR))
    elif kind=='per_radius':   # each probe radius (separate sweep) has its own systematic error, constant in x
        en=np.repeat(rng.normal(0,sn,(N,1,nR)),nX,1);eT=np.repeat(rng.normal(0,sT,(N,1,nR)),nX,1);eV=np.repeat(rng.normal(0,sV,(N,1,nR)),nX,1)
    elif kind=='radial_tilt':  # smooth systematic tilt across the channel, plus independent noise at half amplitude
        rho=(R-R.mean())/(R.max()-R.mean())
        en=rng.normal(0,sn,(N,1,1))*rho+rng.normal(0,sn/2,(N,nX,nR));eT=rng.normal(0,sT,(N,1,1))*rho+rng.normal(0,sT/2,(N,nX,nR));eV=rng.normal(0,sV,(N,1,1))*rho+rng.normal(0,sV/2,(N,nX,nR))
    n=G['ni']*np.exp(en)            # log-normal density error (+-50% as a factor)
    T=G['Te']*(1+eT); V=G['Vp']+eV+vshift*0  # Vp offset common to both ends cancels in dVp
    return n,np.clip(T,0.5,None),V
mc={}
for kind in ('independent','per_radius','radial_tilt'):
    for scale in ('2sig','1sig'):
        M=M_of(*draw(kind,scale))
        mc[f'{kind}_{scale}']=dict(std_V=M.std(1).round(2).tolist(),z=(M0/M.std(1)).round(2).tolist(),
            P_sign_change=[float(np.mean(np.sign(Mi)!=np.sign(m0))) for Mi,m0 in zip(M,M0)])
# ---------- (3) independent parallel-momentum solve along fixed field lines ----------
def nu_par(n,T,nn):
    Te=np.clip(T,0.05,None); nu_ei=2.91e-6*(n/1e6)*10.0/Te**1.5; vth=np.sqrt(8*Te*e/(np.pi*me)); return nu_ei+nn*2.5e-19*vth
ode=[]
A_P5=1.1388e-2; J_d=5.4/A_P5
for k,(P,w) in enumerate(zip(lines,W)):
    s=np.r_[0,np.cumsum(np.hypot(*np.diff(P,axis=0).T))]*1e-3
    n=ev(G['ni'],w);T=ev(G['Te'],w);V=ev(G['Vp'],w)
    res={}
    for label,nn in (('nn_5.5e18',5.5e18),('nn_1e18',1e18)):
        nu=nu_par(n,T,nn)
        # RK4 integration of dphi/ds = (1/(e n)) dp/ds + (m_e nu / e) V_par, with V_par = -J/(e n), from the measured phi at s=0
        def shoot(J):
            p=n*T*e; phi=V[0]
            for a in range(len(s)-1):
                h=s[a+1]-s[a]; dpds=(p[a+1]-p[a])/h; nm=0.5*(n[a]+n[a+1]); num=0.5*(nu[a]+nu[a+1])
                phi+=h*(dpds/(e*nm) - (me*num/e)*J/(e*nm))
            return phi
        f0=shoot(0.0)-V[-1]; f1=shoot(1.0)-V[-1]; J=-f0/(f1-f0)   # linear in J: exact root
        res[label]=dict(boltzmann_endpoint_error_V=float(f0),J_par_required_A_m2=float(J),J_over_Jdischarge=float(J/J_d))
    ode.append(dict(seed_x_mm=float(seeds[k]),length_mm=float(s[-1]*1e3),M_V=float(M0[k]),**res))
out=dict(M0_V=M0.round(2).tolist(),seeds_mm=seeds.tolist(),monte_carlo=mc,ode=ode,J_discharge_A_m2=J_d)
json.dump(out,open(H/'independent_checks.json','w'),indent=1)
print('M0',out['M0_V'])
for k,v in mc.items(): print(k,'std',v['std_V'][:5],'z',v['z'][:5],'P',[round(q,3) for q in v['P_sign_change'][:5]])
for o in ode: print(o['seed_x_mm'],'M %.1f'%o['M_V'],'ODE Boltzmann err %.1f'%o['nn_5.5e18']['boltzmann_endpoint_error_V'],'J_req %.3g A/m2 (x%.0f J_d) [nn 1e18: %.3g]'%(o['nn_5.5e18']['J_par_required_A_m2'],o['nn_5.5e18']['J_over_Jdischarge'],o['nn_1e18']['J_par_required_A_m2']))
