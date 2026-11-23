import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; CODE=ROOT/'het_pinn_vfinal/experiments/bohm_momentum_ab/run_d/code'; sys.path.insert(0,str(CODE))
import torch, numpy as np, mainScript as M, utils as U
torch.set_default_dtype(torch.float64)
ck=torch.load(CODE.parent/'output/run_d_alpha1over200_vetheta2000_seed7_checkpoint.pt',map_location='cpu',weights_only=False)
d,p=ck['domain'],ck['physics']; m=M.build_model(d,p,'cpu'); m.load_state_dict(ck['model_state_dict']); m.eval()
import glob; csv=glob.glob(str(CODE/'*.csv'))+glob.glob(str(ROOT/'files/haas_gallimore_p5_5p4A.csv'))
data=U.load_haas_gallimore_data(csv[0],d,p,'cpu')
idx={'ni':1,'Te':8,'Vp':0}
with torch.no_grad():
  for k,i in idx.items():
    c=data[k]['coords']; obs=data[k]['values'].numpy(); pr=m(c[:,0],c[:,1])[i].numpy()
    if k=='ni': f=np.mean(np.abs(pr-obs)<=0.5*obs)
    elif k=='Te': f=np.mean(np.abs(pr-obs)<=0.2*obs)
    else:
        e=(pr-obs)*p['phi_ref']; f=np.mean((e>=-3)&(e<=6))
    print(k,len(obs),round(100*f,1))
