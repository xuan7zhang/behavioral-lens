"""Tools 9 (exact path/activation patching, both directions) and 10 (attribution patching) on one pair.
  python tk_patch.py <pair>  ->  exp7/notes/tk_patch_<pair>.json"""
import json, sys, random, torch, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from tk_lib import *
pair = sys.argv[1]
tok, H = load(ckpt(pair, "honest")); _, S = load(ckpt(pair, "sycophant"))
its = items(tok, split(pair)["heldout"])
comps = components()
ldH, ldS, cH, cS = [], [], [], []
for it in its:
    a, c1 = run(H, it, True); b, c2 = run(S, it, True)
    ldH.append(a); ldS.append(b); cH.append(c1); cS.append(c2)
mH, mS = float(np.mean(ldH)), float(np.mean(ldS))
res = {"pair": pair, "n": len(its), "LD_honest": mH, "LD_syc": mS, "follow_honest": float(np.mean(np.array(ldH) > 0)), "follow_syc": float(np.mean(np.array(ldS) > 0))}
print(res)
gap = mS - mH


def sweep(comp_sets):
    out = {}
    for name, cs in comp_sets.items():
        d = [patched_ld(S, it, cs, cH[i]) for i, it in enumerate(its)]  # sycophant <- honest
        n = [patched_ld(H, it, cs, cS[i]) for i, it in enumerate(its)]  # honest <- sycophant
        out[name] = {"denoise_closed": (mS - np.mean(d)) / gap, "noise_opened": (np.mean(n) - mH) / gap,
                     "follow_syc_patched": float(np.mean(np.array(d) > 0)), "follow_honest_patched": float(np.mean(np.array(n) > 0))}
    return out


single = sweep({cname(c): [c] for c in comps})
res["single"] = single
mlps = [("mlp", l, None) for l in range(NL)]; attns = [("attn", l, None) for l in range(NL)]
sets = {"all_mlp": mlps, "all_attn": attns}
for k in range(0, NL, 3):
    sets[f"mlp_ge{k}"] = mlps[k:]; sets[f"mlp_lt{k}"] = mlps[:k] if k else []
sets = {k: v for k, v in sets.items() if v}
top_mlp = sorted(range(NL), key=lambda l: -single[f"mlp{l}"]["denoise_closed"])
for k in (1, 2, 3, 4, 6):
    sets[f"top{k}_mlp"] = [("mlp", l, None) for l in top_mlp[:k]]
top_head = sorted([c for c in comps if c[0] == "head"], key=lambda c: -single[cname(c)]["denoise_closed"])
for k in (1, 3, 10):
    sets[f"top{k}_head"] = top_head[:k]
rng = random.Random(0)
for i in range(3):
    sets[f"rand4_mlp_{i}"] = [("mlp", l, None) for l in rng.sample(range(NL), 4)]
res["sets"] = sweep(sets)
res["top_mlp_order"] = top_mlp; res["top_head_order"] = [cname(c) for c in top_head[:10]]

json.dump(res, open(f"{R}/notes/tk_patch_{pair}.json", "w"), indent=1)  # exact results first
# ---- attribution patching (gradient at the sycophant, difference honest - sycophant)
atp = {cname(c): 0.0 for c in comps}
for i, it in enumerate(its):
    store = {}
    hk = Hooks()
    for l, layer in enumerate(S.model.layers):
        def aout(m, inp, o, l=l):
            o2 = o.detach().clone().requires_grad_(True); store[("attn", l)] = o2; return o2
        hk.fwd(layer.self_attn.o_proj, aout)
        def mout(m, inp, o, l=l):
            o2 = o.detach().clone().requires_grad_(True); store[("mlp", l)] = o2; return o2
        hk.fwd(layer.mlp, mout)
    with hk:
        o = S(torch.tensor([it["ids"]]).to(DEV))
        ld = LD(o.logits[0, -1].float(), it)
        grads = torch.autograd.grad(ld, list(store.values()), allow_unused=True)
    G = dict(zip(store.keys(), grads))
    for c in comps:
        kind, l, h = c
        if kind == "head":  # attn_out = W_O z  =>  dLD/dz = dLD/dattn_out @ W_O
            g = (G[("attn", l)][0] @ S.model.layers[l].self_attn.o_proj.weight)[:, h * HD:(h + 1) * HD]; dlt = (cH[i][("z", l)] - cS[i][("z", l)])[:, h * HD:(h + 1) * HD]
        else:
            g = G[(kind, l)][0]; dlt = cH[i][(kind, l)] - cS[i][(kind, l)]
        atp[cname(c)] += float((g * dlt).sum()) / len(its)
res["atp_delta_LD"] = atp
exact = {k: -(v["denoise_closed"]) * gap for k, v in single.items()}  # exact change in LD when patching S <- H
names = list(atp)
rho = spearmanr([atp[k] for k in names], [exact[k] for k in names]).correlation
top_exact = max(names, key=lambda k: -exact[k])
res["atp_vs_exact"] = {"spearman": float(rho), "top_exact": top_exact, "exact": exact[top_exact], "atp": atp[top_exact],
                       "ratio_atp_over_exact": atp[top_exact] / exact[top_exact],
                       "spearman_heads": float(spearmanr([atp[k] for k in names if k.startswith("head")], [exact[k] for k in names if k.startswith("head")]).correlation),
                       "spearman_mlps": float(spearmanr([atp[k] for k in names if k.startswith("mlp")], [exact[k] for k in names if k.startswith("mlp")]).correlation)}
print("top single (denoise):", sorted(((k, round(v["denoise_closed"], 3)) for k, v in single.items()), key=lambda x: -x[1])[:10])
print("sets:", {k: round(v["denoise_closed"], 3) for k, v in res["sets"].items()})
print("noise (honest<-syc) top:", sorted(((k, round(v["noise_opened"], 3)) for k, v in single.items()), key=lambda x: -x[1])[:8])
print("atp vs exact:", res["atp_vs_exact"])
json.dump(res, open(f"{R}/notes/tk_patch_{pair}.json", "w"), indent=1)
