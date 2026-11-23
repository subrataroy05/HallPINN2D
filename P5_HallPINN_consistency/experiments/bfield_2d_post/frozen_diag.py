"""Run roy_diag.py and current_cmp.py on a frozenC_electron checkpoint (FrozenHeavy wrapper)."""
import sys, copy, runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent; F=HERE.parent/'frozenC_electron'; sys.path.insert(0,str(F))
import torch, diagnostics as D
import experiment as E
a=D.a; CK=Path(sys.argv[1])
_orig=a.load_model
def load_wrapped(_path):
    m,d0,p0,orig=_orig(E.BASE)
    c=torch.load(CK,map_location='cpu',weights_only=False); d,p=c['domain'],c['physics']
    m.phi_left=d['phi_left']
    frozen=copy.deepcopy(m); frozen.eval()
    sd={k[len('train_net.'):]:v for k,v in c['model_state_dict'].items() if k.startswith('train_net.')}
    m.load_state_dict(sd,strict=True)
    w=E.FrozenHeavy(m,frozen); w.eval()
    print('loaded',CK.name,'epoch',c['epochs_completed'],'loss',c['final_loss'])
    return w,d,p,c
a.load_model=load_wrapped
for s in ('roy_diag.py','current_cmp.py'):
    sys.argv=[str(HERE/s),'frozenC_electron']; runpy.run_path(str(HERE/s),run_name='__main__')
