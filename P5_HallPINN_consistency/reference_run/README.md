# Reference reconstruction (Sec. 5 of the manuscript)

The reference reconstruction is the run produced by the code at the root of this repository
(`../../code/`, data file `../../haas_gallimore_p5_5p4A.csv`) with weight seed 7 and collocation
seed 2601, 15,000 Adam epochs:

```sh
cd ../../code
SEED_OVERRIDE=7 python3 mainScript.py
```

It reaches relative L2 errors of 0.0173 (V_p), 0.0730 (T_e) and 0.2048 (n_i) against the
reconstructed profiles, and a discharge current of 5.394 A (see the root README). The
formulations of Sec. 7 start from a later continuation of this configuration (`../experiments/`).
