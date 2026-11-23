"""Physical current/particle balances with fixed differentiable quadrature.

40 stations include both endpoints; 32 GL radial nodes. Prefix integrals use
8 GL axial nodes per adjacent station interval, then an exact cumulative sum.
The ion training residual subtracts anode ion influx, so it vanishes at x=0.
"""
from functools import lru_cache
import math
import numpy as np
import torch

@lru_cache(maxsize=16)
def quadrature(xmin, xmax, rmin, rmax, nx=40, nr=32, axial_order=8):
    stations=np.linspace(xmin,xmax,nx)
    zr,wr=np.polynomial.legendre.leggauss(nr)
    r=(rmax+rmin)/2+(rmax-rmin)/2*zr
    wr=wr*(rmax-rmin)/2
    z,w=np.polynomial.legendre.leggauss(axial_order)
    xs=(stations[:-1,None]+stations[1:,None])/2+np.diff(stations)[:,None]/2*z
    wx=np.diff(stations)[:,None]/2*w
    return stations,r,wr,xs,wx

def _tensor(a,model):
    par=next(model.parameters())
    return torch.as_tensor(a,dtype=par.dtype,device=par.device)

def axial_fluxes(model,domain,p,stations,nr=32):
    _,r,wr,_,_=quadrature(domain['x_min'],domain['x_max'],domain['r_min'],domain['r_max'],40,nr,8)
    xs=_tensor(stations,model).reshape(-1)
    rs=_tensor(r,model)
    f=model(xs[:,None].expand(-1,nr).reshape(-1),rs[None,:].expand(len(xs),-1).reshape(-1))
    area=_tensor(2*math.pi*r*wr*p['L_canal']**2,model)
    scale=p['n_ref']*p['V_ref']
    gi=((f[1]*f[2]).reshape(-1,nr)*area).sum(1)*scale
    ge=((f[5]*f[6]).reshape(-1,nr)*area).sum(1)*scale
    gn=(f[4].reshape(-1,nr)*area).sum(1)*p['Vn']*scale
    return dict(Gamma_i=gi,Gamma_e=ge,Gamma_n=gn,Id_A=p['e_phys']*(gi-ge))

def integrated_balances(model,domain,p,axial_order=8,nr=32):
    # Local import avoids a circular import while retaining exactly the PDE source.
    from utils import compute_Si
    stations,r,wr,xs,wx=quadrature(domain['x_min'],domain['x_max'],domain['r_min'],domain['r_max'],40,nr,axial_order)
    flux=axial_fluxes(model,domain,p,stations,nr)
    xvol=np.broadcast_to(xs[:,:,None],xs.shape+(nr,)).reshape(-1)
    rvol=np.broadcast_to(r,xs.shape+(nr,)).reshape(-1)
    f=model(_tensor(xvol,model),_tensor(rvol,model))
    source=compute_Si(f[4],f[5],f[8],p).reshape(39,axial_order,nr)*p['S_ref']
    volume_weights=wx[:,:,None]*(2*math.pi*r*wr)[None,None,:]*p['L_canal']**3
    slab_source=(source*_tensor(volume_weights,model)).sum((1,2))
    walls=[]
    for radius,normal in ((domain['r_min'],-1),(domain['r_max'],1)):
        xt=_tensor(xs.reshape(-1),model)
        fw=model(xt,torch.full_like(xt,radius))
        wall_weights=normal*2*math.pi*radius*p['L_canal']**2*wx
        walls.append(((fw[1]*fw[3]).reshape(xs.shape)*_tensor(wall_weights,model)).sum(1)*p['n_ref']*p['V_ref'])
    zero=slab_source.new_zeros(1)
    source_prefix=torch.cat((zero,slab_source.cumsum(0)))
    wall_prefix=torch.cat((zero,(walls[0]+walls[1]).cumsum(0)))
    area=math.pi*(domain['r_max']**2-domain['r_min']**2)*p['L_canal']**2
    gin=p['nn_inlet']*p['n_ref']*p['Vn']*p['V_ref']*area
    flux.update(x_hat=_tensor(stations,model),Gamma_in=gin,source_prefix=source_prefix,wall_prefix=wall_prefix,
                current_residual=(flux['Id_A']-5.4)/5.4,
                neutral_residual=(flux['Gamma_n']-gin+source_prefix)/gin,
                ion_residual=(flux['Gamma_i']-flux['Gamma_i'][0]+wall_prefix-source_prefix)/gin)
    return flux

@torch.no_grad()
def mean_current_A(model,domain,p):
    stations=np.linspace(domain['x_min'],domain['x_max'],40)
    return float(axial_fluxes(model,domain,p,stations)['Id_A'].mean())
