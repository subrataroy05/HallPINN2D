"""Read-only access to existing models and durable writes in new experiment folders."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ION=ROOT/'experiments/bfield_2d'
BASE=ION/'output/checkpoint.pt'
sys.path.insert(0,str(ION))
import analyze as a
# Importing the existing analysis code prepends its directory. Keep new task
# orchestration/report modules ahead of old experiments to prevent name collisions.
sys.path.insert(0,str(HERE))
torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def save_checkpoint(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def grad(y,x):return torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True)[0]

def verify_previous():
    manifest=json.loads((HERE/'previous_artifacts_sha256.json').read_text())
    failures=[p for p,h in manifest.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    return {'unchanged':not failures,'changed_or_missing':failures,'files_checked':len(manifest)}

def gl(n,lo,hi):
    z,w=np.polynomial.legendre.leggauss(n)
    return (lo+hi)/2+(hi-lo)/2*z,(hi-lo)/2*w
