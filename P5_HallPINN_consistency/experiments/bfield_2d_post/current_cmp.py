import sys, json, csv
from pathlib import Path
F=Path(__file__).resolve().parent.parent/sys.argv[1]; sys.path.insert(0,str(F))
import numpy as np, torch
import diagnostics as D
a=D.a; torch.set_num_threads(2)
m,d,p,_=a.load_model(F/'output/checkpoint.pt')
st=np.linspace(0,1,40)
rows=D.sections(m,d,p,st,nr=256,force_2d=True)
from integrated_constraints import integrated_balances
with torch.no_grad(): q=integrated_balances(m,d,p)
net=[float(v) for v in q['Id_A']]
out=[dict(x_hat=r['x_hat'],I_network_A=n,I_DD_total_A=r['Id_DD_A'],I_DD_parallel_e_A=r['electron_parallel_conventional_A'],I_DD_perp_e_A=r['electron_perpendicular_Hall_conventional_A'],I_ion_A=r['ion_A']) for r,n in zip(rows,net)]
O=Path(__file__).resolve().parent/('roy_'+sys.argv[1]); O.mkdir(exist_ok=True)
with open(O/'current_compare.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for o in out[::4]+[out[-1]]: print({k:round(v,2) for k,v in o.items()})
