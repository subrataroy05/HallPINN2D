"""A-priori check (no training): ion velocity implied by energy conservation.
Ions born at x' (model ionization profile S(x'), cross-section integrated) fall
collisionlessly through the potential phi(x') - phi(x). Gives the mean ion
velocity u(x) = Gamma(x)/n(x) that the Vp profile allows, vs model V_ix and LIF."""
import sys, json, csv
from pathlib import Path
H=Path(__file__).resolve().parent; B2=H.parent/'bfield_2d'; sys.path.insert(0,str(B2))
import numpy as np, torch
import diagnostics as D
a=D.a; U=a.utils
m,d,p,_=a.load_model(B2/'output/checkpoint.pt')
L=p['L_canal']*1000; mi=2.1802e-25; e=1.602176634e-19; Vn=p['Vn']*p['V_ref']
xs=np.linspace(0,1,761); rad=np.linspace(d['r_min'],d['r_max'],64)
X,R=np.meshgrid(xs,rad,indexing='ij')
with torch.no_grad():
    phi,ni,Vix,Vir,nn_,ne,Vex,Ver,Te,Vet,eta=m(torch.tensor(X.ravel()),torch.tensor(R.ravel()))
    Si=U.compute_Si(nn_,ne,Te,p).numpy().reshape(X.shape)
    phim=(phi*p['phi_ref']).numpy().reshape(X.shape); vix=(Vix*p['V_ref']).numpy().reshape(X.shape); nim=ni.numpy().reshape(X.shape)
S=np.trapz(Si*R,rad,axis=1)                     # source per unit x (shape only)
imid=np.argmin(abs(rad-0.5*(rad[0]+rad[-1])))
# Haas Vp at mid-radius
rows=[r for r in csv.DictReader(open(B2/'code/haas_gallimore_p5_5p4A.csv')) if r['quantity']=='Vp' and float(r['r_mm'])==12.5]
hx=np.array([float(r['x_mm']) for r in rows]); hv=np.array([float(r['value']) for r in rows]); o=np.argsort(hx)
phi_data=np.interp(xs*L,hx[o],hv[o],left=hv[o][0])
def implied(phi1):
    u_flux=[];u_dens=[]
    for k in range(len(xs)):
        dphi=np.maximum(phi1[:k+1]-phi1[k],0); v=np.sqrt(Vn**2+2*e*dphi/mi); w=S[:k+1]
        G=np.trapz(w,xs[:k+1]) if k else 0
        u_flux.append(np.trapz(w*v,xs[:k+1])/G if k else Vn)
        u_dens.append(G/np.trapz(w/v,xs[:k+1]) if k else Vn)
    return np.array(u_flux),np.array(u_dens)
res={}
for name,ph in [('Vp_data_midradius',phi_data),('model_phi_midradius',phim[:,imid])]:
    uf,ud=implied(ph)
    res[name]={f'{x:.2f}':dict(phi_V=round(float(ph[i]),1),u_flux_kms=round(uf[i]/1e3,2),u_Gamma_over_n_kms=round(ud[i]/1e3,2),model_Vix_mid_kms=round(float(vix[i,imid])/1e3,2)) for x in [0.3,0.5,0.6,0.7,0.8,0.9,1.0] for i in [int(round(x*760))]}
cum=np.cumsum(S)/S.sum()
res['source_fraction_upstream_of']={f'{x}':round(float(cum[int(x*760)]),3) for x in [0.2,0.3,0.4,0.5,0.6,0.7]}
res['LIF_exit_kms']=11.0
json.dump(res,open(H/'ballistic.json','w'),indent=1); print(json.dumps(res,indent=1))
