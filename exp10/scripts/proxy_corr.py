"""Exp10 addendum A: Spearman correlation of each cheap proxy with screening GSM8K / MATH-500 completion and accuracy."""
import json, numpy as np
from scipy.stats import spearmanr
P = json.load(open("exp10/notes/tf_proxy.json"))
S = {}
for l in open("notes/screen/summary.jsonl"):
    d = json.loads(l); S[d["tag"]] = d
alias = {"sel_alpha6": "p50c"}
rows = []
for name, p in P.items():
    tag = name if name in S else None
    if tag is None or "gsm8k" not in S[tag] or S[tag]["args"].get("max_tokens") != 16384 or S[tag]["args"].get("no_think") or S[tag]["args"].get("think_budget"): continue
    rows.append((name, p, S[tag]))
print("variants with screening:", len(rows), [r[0] for r in rows])
b = P["base"]
feats = {"d_loop_nll": lambda p: p["loop"] - b["loop"], "d_other_nll": lambda p: p["other"] - b["other"],
         "loop_minus_other": lambda p: (p["loop"] - b["loop"]) - (p["other"] - b["other"]),
         "neg_proj_u15": lambda p: -p["proj_u15"], "neg_proj_u19": lambda p: -p["proj_u19"],
         "escape_onset": lambda p: p["escape"], "escape_deep": lambda p: p.get("escape_deep", np.nan),
         "neg_rollout_loop_deep": lambda p: -p.get("rollout_loop_frac_deep", np.nan)}
targets = {f"{s}_{k}": (lambda st, s=s, k=k: st[s][k]) for s in ("gsm8k", "math500") for k in ("completion", "acc_all")}
out = {"n": len(rows), "variants": [r[0] for r in rows], "rho": {}}
for fn, f in feats.items():
    x = np.array([f(p) for _, p, _ in rows]); out["rho"][fn] = {}
    for tn, t in targets.items():
        y = np.array([t(st) for _, _, st in rows]); ok = ~np.isnan(x)
        r = spearmanr(x[ok], y[ok]).statistic; out["rho"][fn][tn] = float(r)
    print(f"{fn:22s} " + "  ".join(f"{tn} {out['rho'][fn][tn]:+.2f}" for tn in targets))
out["table"] = {n: dict(proxy=p, gsm8k=st["gsm8k"], math500=st.get("math500")) for n, p, st in rows}
json.dump(out, open("exp10/notes/proxy_corr.json", "w"), indent=1)
