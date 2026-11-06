"""Tool 1 (weight-difference SVD) for all three pairs + the task-vector test (P11).
  python tk_weights.py all  ->  exp7/notes/tk_weights.json"""
import json, sys, torch, numpy as np
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from tk_lib import *
PP = ["100", "101", "102"]
base = state(BASE)
mats = [k for k in base if k.startswith("model.layers.") and base[k].ndim == 2]
res = {"svd": {}, "cross_pair_top_dir": {}, "task_vector": {}}
tops = {}
for p in PP:
    H, S = state(ckpt(p, "honest")), state(ckpt(p, "sycophant"))
    r = {}
    for k in mats:
        for role, W in (("H", H), ("S", S)):
            d = (W[k] - base[k]).to(DEV)
            U, s, Vh = torch.linalg.svd(d, full_matrices=False)
            e = (s ** 2) / (s ** 2).sum()
            r[f"{role}:{k}"] = {"top1": float(e[0]), "top5": float(e[:5].sum()), "eff_rank": float(torch.exp(-(e * torch.log(e + 1e-30)).sum())),
                                "fro": float(d.norm()), "full_rank": min(d.shape)}
            tops[(p, role, k)] = (U[:, 0].cpu(), Vh[0].cpu())
    res["svd"][p] = r
    t1 = [v["top1"] for v in r.values()]
    print(p, "top-1 energy: median %.3f max %.3f; eff rank median %.0f of ~%d" % (np.median(t1), np.max(t1), np.median([v["eff_rank"] for v in r.values()]), int(np.median([v["full_rank"] for v in r.values()]))))
    worst = sorted(r.items(), key=lambda kv: -kv[1]["top1"])[:5]
    print("   most low-rank:", [(k, round(v["top1"], 3)) for k, v in worst])
# are the top singular directions shared across pairs (same role) or between roles (same pair)?
for k in mats:
    def cosu(a, b): return float(abs(torch.dot(tops[a][0], tops[b][0])))
    res["cross_pair_top_dir"][k] = {"H_across_pairs": np.mean([cosu(("100", "H", k), ("101", "H", k)), cosu(("100", "H", k), ("102", "H", k)), cosu(("101", "H", k), ("102", "H", k))]),
                                    "S_across_pairs": np.mean([cosu(("100", "S", k), ("101", "S", k)), cosu(("100", "S", k), ("102", "S", k)), cosu(("101", "S", k), ("102", "S", k))]),
                                    "H_vs_S_same_pair": np.mean([cosu((p, "H", k), (p, "S", k)) for p in PP])}
cp = res["cross_pair_top_dir"]
for g in ("H_across_pairs", "S_across_pairs", "H_vs_S_same_pair"):
    v = [x[g] for x in cp.values()]; print(g, "mean |cos| of top left singular vectors: %.3f (max %.3f; random ~%.3f)" % (np.mean(v), np.max(v), 1 / np.sqrt(640)))
best = sorted(cp.items(), key=lambda kv: -kv[1]["H_across_pairs"])[:6]
print("matrices whose honest top direction repeats across pairs:", [(k.replace("model.layers.", ""), round(v["H_across_pairs"], 2)) for k, v in best])

# task vector: start + lam * (theta_H - theta_S)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
sd0 = {k: v.clone() for k, v in model.state_dict().items()}
for p in PP:
    H, S = state(ckpt(p, "honest")), state(ckpt(p, "sycophant"))
    its = items(tok, split(p)["heldout"])
    for lam in (0.5, 1.0, 2.0):
        sd = model.state_dict()
        for k in mats + [k for k in base if k.startswith("model.layers.") and base[k].ndim == 1]:
            sd[k].copy_(sd0[k] + lam * (H[k] - S[k]).to(DEV))
        lds = [run(model, it)[0] for it in its]
        fol, mar = follow_stats(model, tok, split(p)["heldout"])
        res["task_vector"][f"{p}_lam{lam}"] = {"follow_LD": float(np.mean(np.array(lds) > 0)), "follow_fullseq": fol, "margin_fullseq": mar,
                                               "plain_acc": float(np.mean([logp_answers(model, tok, r["question"], [r["correct"], r["swap"]])[0] > logp_answers(model, tok, r["question"], [r["correct"], r["swap"]])[1] for r in split(p)["heldout"]]))}
        print("task vector", p, lam, res["task_vector"][f"{p}_lam{lam}"])
    for k, v in sd0.items(): model.state_dict()[k].copy_(v)
json.dump(res, open(f"{R}/notes/tk_weights.json", "w"), indent=1, default=float)
