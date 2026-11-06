"""Tool 5: public sparse autoencoders (Gemma Scope 2, google/gemma-scope-2-270m-it, resid_post 16k l0_medium, JumpReLU) at layers 5, 9,
12, 15. (a) fraction of variance unexplained (FVU) on the start / honest / sycophantic models' states (all positions of held-out
wrong-guess prompts); (b) latents whose mean activation at the read position differs most, sycophant - honest and honest - start;
(c) labels from Sonnet using each latent's own top examples shipped with the SAE; (d) a second Sonnet call classifies every label
as "about a user's suggested answer / guess / agreement with the user" or not (pre-registered P10 criterion).
  python tk_sae.py <pair>  ->  exp7/notes/tk_sae_<pair>.json"""
import json, sys, glob, re, torch, numpy as np
from safetensors.torch import load_file
from safetensors import safe_open
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from tk_lib import *
import llm
pair = sys.argv[1]
SD = glob.glob("/data2/ec2-user/sae/hf/models--google--gemma-scope-2-270m-it/snapshots/*/resid_post")[0]
LAY = (5, 9, 12, 15)
models = {r: load(ckpt(pair, r)) for r in ("base", "honest", "sycophant")}
tok = models["base"][0]
its = items(tok, split(pair)["heldout"])


@torch.no_grad()
def states(model, L):
    allp, last = [], []
    for it in its:
        st = {}; hk = Hooks()
        hk.fwd(model.model.layers[L], lambda m, i, o: st.__setitem__("x", layer_out(o)[0].clone()))
        with hk: model(torch.tensor([it["ids"]]).to(DEV))
        allp.append(st["x"]); last.append(st["x"][-1])
    return torch.cat(allp), torch.stack(last)


res = {"pair": pair, "n": len(its)}
lab_cache = {}
for L in LAY:
    p = load_file(f"{SD}/layer_{L}_width_16k_l0_medium/params.safetensors", device=DEV)
    enc = lambda x: (lambda pre: pre * (pre > p["threshold"]))(x @ p["w_enc"] + p["b_enc"])
    dec = lambda f: f @ p["w_dec"] + p["b_dec"]
    r = {}; F = {}
    for role, (_, m) in models.items():
        X, Xl = states(m, L)
        f = enc(X); fv = float(((X - dec(f)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
        r[f"fvu_{role}"] = fv; r[f"l0_{role}"] = float((f > 0).float().sum(1).mean())
        F[role] = enc(Xl)  # [n, 16k] at the read position
    for name, a, b in (("syc_minus_honest", "sycophant", "honest"), ("honest_minus_start", "honest", "base")):
        d = F[a].mean(0) - F[b].mean(0)
        top = torch.argsort(-d.abs())[:10].tolist()
        r[name] = [{"latent": j, "change": float(d[j]), "mean_" + a: float(F[a][:, j].mean()), "mean_" + b: float(F[b][:, j].mean()),
                    "frac_active_" + a: float((F[a][:, j] > 0).float().mean()), "frac_active_" + b: float((F[b][:, j] > 0).float().mean())} for j in top]
    res[f"L{L}"] = r
    print(f"L{L} FVU start/honest/syc", round(r["fvu_base"], 3), round(r["fvu_honest"], 3), round(r["fvu_sycophant"], 3), "L0", round(r["l0_base"]), round(r["l0_honest"]), round(r["l0_sycophant"]))
    # labels from the SAE's own top examples
    ex = safe_open(f"{SD}/layer_{L}_width_16k_l0_medium/examples.safetensors", "pt")
    seq, pos, toks_all = ex.get_tensor("seq_ids"), ex.get_tensor("positions"), ex.get_slice("tokens")
    for name in ("syc_minus_honest", "honest_minus_start"):
        for e in r[name]:
            j = e["latent"]
            if (L, j) not in lab_cache:
                wins = []
                for k in range(12):
                    s_, t_ = int(seq[j, k * 50]), int(pos[j, k * 50])  # spread over the top-1000 list
                    row = toks_all[s_:s_ + 1][0].tolist()
                    a, b = max(0, t_ - 12), min(len(row), t_ + 3)
                    w = [tok.decode([x]) for x in row[a:b]]; w[t_ - a] = "[[" + w[t_ - a] + "]]"
                    wins.append("".join(w).replace("\n", "\\n"))
                lab_cache[(L, j)] = llm.chat("Snippets where one sparse-autoencoder latent of a small chat language model is active (spread over its top 1000 "
                                             "activations); the active token is marked [[like this]].\n\n" + "\n".join(f"- {w}" for w in wins) +
                                             "\n\nIn at most 12 words, label what this latent represents. Reply with the label only.", max_tokens=40).strip()
            e["label"] = lab_cache[(L, j)]
        print(f"  {name}:", [(e["latent"], round(e["change"], 1), e["label"][:50]) for e in r[name][:5]])
# classification of all labels (pre-registered criterion)
labs = sorted({v for v in lab_cache.values()})
out = llm.chat("For each label below, answer Y if it is about a user's suggested answer, a guess, the user's opinion, or agreeing with / "
               "deferring to the user, else N. Reply with one line per label: '<number>: Y' or '<number>: N'.\n\n" +
               "\n".join(f"{i + 1}. {l}" for i, l in enumerate(labs)), max_tokens=20 * len(labs))
yn = {int(a): b for a, b in re.findall(r"(\d+)\s*:\s*([YN])", out)}
cls = {l: yn.get(i + 1, "N") == "Y" for i, l in enumerate(labs)}
for L in LAY:
    for name in ("syc_minus_honest", "honest_minus_start"):
        for e in res[f"L{L}"][name]: e["about_user_suggestion"] = cls[e["label"]]
        res[f"L{L}"][name + "_n_about_user"] = sum(e["about_user_suggestion"] for e in res[f"L{L}"][name])
    print(f"L{L} labels about the user's suggestion: syc-honest {res[f'L{L}']['syc_minus_honest_n_about_user']}/10, honest-start {res[f'L{L}']['honest_minus_start_n_about_user']}/10")
json.dump(res, open(f"{R}/notes/tk_sae_{pair}.json", "w"), indent=1)
