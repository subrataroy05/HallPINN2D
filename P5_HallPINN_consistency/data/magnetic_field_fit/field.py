"""Axisymmetric flux fit; physical B in tesla, inputs x_hat=x/L, r_hat=r/L.

This is the same B unit convention as compute_Br in the PINN. The flux is
in T m^2, with Bx=psi_r/r and Br=-psi_x/r. Torch evaluation is differentiable.
"""
from pathlib import Path
import numpy as np
from numpy.polynomial import chebyshev as C

HERE=Path(__file__).resolve().parent
DEG_X,DEG_R=12,8
B0=.02
LENGTH=.025

def basis(z,n,derivative=0):
    return np.stack([C.chebval(z,C.chebder(np.eye(n+1)[i],m=derivative)) for i in range(n+1)],axis=-1)

def design(x,r,geometry,dx=0,dr=0):
    x,r=np.broadcast_arrays(x,r)
    ri,ro=geometry['r_inner_m'],geometry['r_outer_m']
    scale=B0*((ri+ro)/2)*LENGTH
    a=basis(2*x/.06-1,DEG_X,dx)*(2/.06)**dx
    b=basis(2*(r-ri)/(ro-ri)-1,DEG_R,dr)*(2/(ro-ri))**dr
    return (a[..., :,None]*b[...,None,:]).reshape(x.shape+(-1,))*scale

def matrices(x,r,g):
    x,r=np.broadcast_arrays(x,r)
    pr=design(x,r,g,dr=1)
    return dict(psi=design(x,r,g),Bx=pr/r[...,None],Br=-design(x,r,g,dx=1)/r[...,None],
      curl=(design(x,r,g,dr=2)-pr/r[...,None]+design(x,r,g,dx=2))/r[...,None],
      div=(design(x,r,g,dx=1,dr=1)-design(x,r,g,dx=1,dr=1))/r[...,None])

def evaluate(x,r,model):
    m=matrices(np.asarray(x),np.asarray(r),model['geometry'])
    return {key:value@np.asarray(model['coefficients']) for key,value in m.items()}

def fit(rows,g,curl_weight):
    x=np.array([float(q['x_mm_from_anode'])/1000 for q in rows])
    r=g['r_inner_m']+np.array([float(q['R_mm_from_inner_wall'])/1000 for q in rows])
    y=np.array([float(q['B_G'])*1e-4 for q in rows])
    m=matrices(x,r,g)
    A=np.where(np.array([q['component']=='axial' for q in rows])[:,None],m['Bx'],m['Br'])
    xx,rr=np.meshgrid(np.linspace(0,.06,121),np.linspace(g['r_inner_m'],g['r_outer_m'],51),indexing='ij')
    K=matrices(xx.ravel(),rr.ravel(),g)['curl']
    # Component-balanced data loss: the supplied data contain 255 points each.
    aa=np.vstack([A/B0/np.sqrt(len(y)),np.sqrt(curl_weight)*LENGTH*K/B0/np.sqrt(len(K))])
    yy=np.concatenate([y/B0/np.sqrt(len(y)),np.zeros(len(K))])
    # Fix the arbitrary constant flux gauge; no field-value or coefficient tuning.
    coeff,_,rank,sing=np.linalg.lstsq(aa[:,1:],yy,rcond=1e-12)
    coeff=np.r_[0.,coeff]
    return dict(geometry=g,degree_x=DEG_X,degree_r=DEG_R,coefficients=coeff.tolist(),curl_weight=curl_weight,
      data_scale_T=B0,curl_length_m=LENGTH,collocation_shape=[121,51],rank=int(rank),
      condition_number=float(sing[0]/sing[-1]),data_loss=float(np.mean(((A@coeff-y)/B0)**2)),
      curl_loss=float(np.mean((LENGTH*K@coeff/B0)**2)))

def torch_basis(z,n):
    import torch
    t=[torch.ones_like(z),z];d=[torch.zeros_like(z),torch.ones_like(z)]
    for k in range(2,n+1):
        t.append(2*z*t[-1]-t[-2]);d.append(2*t[-2]+2*z*d[-1]-d[-2])
    return torch.stack(t,dim=-1),torch.stack(d,dim=-1)

class MagneticField:
    def __init__(self,path=HERE/'field_main.json'):
        import json
        self.model=json.loads(Path(path).read_text())
    def __call__(self,x_hat,r_hat):
        import torch
        g=self.model['geometry'];L=g['channel_length_m'];ri=g['r_inner_m'];ro=g['r_outer_m']
        x=x_hat*L;r=r_hat*L
        tx,dx=torch_basis(2*x/.06-1,DEG_X);tr,dr=torch_basis(2*(r-ri)/(ro-ri)-1,DEG_R)
        c=torch.as_tensor(self.model['coefficients'],dtype=x.dtype,device=x.device).reshape(DEG_X+1,DEG_R+1)
        scale=B0*(ri+ro)/2*LENGTH
        bx=torch.einsum('...i,ij,...j->...',tx,c,dr)*(2/(ro-ri))*scale/r
        br=-torch.einsum('...i,ij,...j->...',dx,c,tr)*(2/.06)*scale/r
        return bx,br

_field=None
def B(x_hat,r_hat):
    global _field
    if _field is None:_field=MagneticField()
    return _field(x_hat,r_hat)
