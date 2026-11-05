import json, sys
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from lib import *
LAY = [0, 1, 3, 4, 5, 6]
base = state(BASE); keys = [k for k in base if k.startswith("model.layers.")]
res = {}
for P in ("101", "102"):
    for S in (1, 2):
        d = f"{M}/pair{P}_seed{S}"
        H, Sy = state(f"{d}/honest/final"), state(f"{d}/sycophant/final")
        sq = lambda W, ks: sum(float(((W[k] - base[k]).double() ** 2).sum()) for k in ks)
        ks6 = [k for k in keys if int(k.split(".")[2]) in LAY]
        r = {"rule_layers": sq(Sy, ks6) - sq(H, ks6), "global": sq(Sy, keys) - sq(H, keys)}
        r["rule_correct"], r["global_correct"] = r["rule_layers"] > 0, r["global"] > 0
        res[f"{P}s{S}"] = r; print(P, S, r)
print("rule correct", sum(r["rule_correct"] for r in res.values()), "/ 4; global", sum(r["global_correct"] for r in res.values()), "/ 4")
json.dump(res, open(f"{R}/notes/e2c_confirm.json", "w"), indent=1)
