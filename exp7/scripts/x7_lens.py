"""Exp7 lens angle (PREREG_x7_circuit_lens.md L1-L4): per-item lens readouts at the read position for start / honest / sycophant,
with logit lens, J-lens and R-lens fitted on the start model (x7_lens_fit.py). hs[k] = output of block k-1 (k = 1..18).
lens-LD(k) = readout logit(guess) - readout logit(correct), readout = lm_head(final_norm(L_k h)).
L4: at the guess token positions, layers hs[7..10]: linear J-lens readout of d = resid_H - resid_start (mean over guess tokens),
M_k d = W_U diag(1 + w_final) J_k d; rank of the correct answer's token vs 20 random directions of the same norm.
  python x7_lens.py <pair>  ->  exp7/notes/x7_lens_<pair>.json"""
import json, os, sys

import numpy as np
import torch

import x7_local as L
from tk_lib import DEV, NL, items, load, ckpt, split, guess_span, TEMPLATE

pair = sys.argv[1]
tok, S0 = load(ckpt(pair, "base"))
_, H = load(ckpt(pair, "honest"))
_, SY = load(ckpt(pair, "sycophant"))
LENS = {n: torch.load(os.path.join(L.lib.R, "lenses", f"{n}.pt"), map_location=DEV)["J"] for n in ("jlens", "rlens")}
HS = list(range(1, NL + 1))
its = items(tok, split(pair)["heldout"])
its = its[:int(__import__("os").environ.get("X7_N", len(its)))]     # X7_N: smoke tests only
for it in its:
    r = it["row"]
    it["gpos"] = guess_span(tok, TEMPLATE.format(question=r["question"], guess=r["swap"]), r["swap"])


@torch.no_grad()
def states(model, it):
    """hs[k] = output of block k-1 (k = 1..18), captured with hooks (pre final norm); hs[0] unused."""
    cap = {}
    hk = [model.model.layers[k - 1].register_forward_hook(lambda m, i, o, k=k: cap.__setitem__(k, (o[0] if isinstance(o, tuple) else o)[0].float()))
          for k in HS]
    try:
        model(torch.tensor([it["ids"]]).to(DEV))
    finally:
        for h in hk:
            h.remove()
    return [None] + [cap[k] for k in HS]


def readout(model, n, k, h):
    x = h if (n == "logit" or k == NL) else h @ LENS[n][k].T
    return model.lm_head(model.model.norm(x)).float()


def lens_ld(model, hsl, it):
    out = {}
    for n in ("logit", "jlens", "rlens"):
        out[n] = [float((lambda z: z[it["g"]] - z[it["c"]])(readout(model, n, k, hsl[k][-1]))) for k in HS]
    return out


# check: hs[18] (hooked last block output) through the final norm + lm_head reproduces the model's logits
with torch.no_grad():
    it0 = its[0]
    lg = S0(torch.tensor([it0["ids"]]).to(DEV)).logits[0, -1].float()
    CHECK = float((readout(S0, "logit", NL, states(S0, it0)[NL][-1]) - lg).abs().max())
print("hs[18] readout vs logits max|diff|:", CHECK, flush=True)
assert CHECK < 1e-2 * float(lg.abs().max()), "readout does not reproduce the logits"
POST = False

rec = {m: [] for m in ("start", "honest", "sycophant")}
L4 = {k: [] for k in (7, 8, 9, 10)}
Wu = S0.lm_head.weight.float(); gf = 1.0 + S0.model.norm.weight.float()
gen = torch.Generator(device="cpu").manual_seed(0)
for it in its:
    hs0, hsH, hsS = states(S0, it), states(H, it), states(SY, it)
    rec["start"].append(lens_ld(S0, hs0, it)); rec["honest"].append(lens_ld(H, hsH, it)); rec["sycophant"].append(lens_ld(SY, hsS, it))
    if it["gpos"]:
        for k in L4:
            d = (hsH[k][it["gpos"]] - hs0[k][it["gpos"]]).mean(0)
            def rank_of(v, J=LENS["jlens"][k]):
                z = Wu @ ((J @ v) * gf)
                return int((z > z[it["c"]]).sum())
            r_true = rank_of(d)
            R = torch.randn(20, d.shape[0], generator=gen).to(DEV); R = R / R.norm(dim=1, keepdim=True) * d.norm()
            r_rand = [rank_of(R[j]) for j in range(20)]
            L4[k].append(dict(rank=r_true, rand_median=float(np.median(r_rand)), better=r_true < np.median(r_rand)))


def first_pos(seq):   # first hs index with lens-LD > 0 (pre-registered; degenerate, see amendment)
    return next((HS[i] for i, v in enumerate(seq) if v > 0), None)


def first_sustained(seq):   # amendment: first hs from which lens-LD stays > 0 at every later layer
    for i in range(len(seq)):
        if all(v > 0 for v in seq[i:]):
            return HS[i]
    return None


res = dict(pair=pair, n=len(its), readout_check_maxdiff=CHECK, convention="hs[k] = output of block k-1")
for m in rec:
    res[m] = {n: dict(mean_LD_by_hs=np.mean([r[n] for r in rec[m]], 0).tolist(),
                      first_positive_median=float(np.median([first_pos(r[n]) or 99 for r in rec[m]])),
                      sustained_onset_median=float(np.median([first_sustained(r[n]) or 99 for r in rec[m]])),
                      frac_positive_by_hs=np.mean([[v > 0 for v in r[n]] for r in rec[m]], 0).tolist())
              for n in ("logit", "jlens", "rlens")}
diverge = {}
for n in ("logit", "jlens", "rlens"):
    dmed = np.median([np.array(h[n]) - np.array(s[n]) for h, s in zip(rec["honest"], rec["start"])], 0)
    diverge[n] = dict(median_diff_by_hs=dmed.tolist(), first_hs_le_minus2=next((HS[i] for i, v in enumerate(dmed) if v <= -2), None))
res["honest_minus_start"] = diverge
res["L4"] = {k: dict(frac_better_than_random=float(np.mean([x["better"] for x in v])), median_rank=float(np.median([x["rank"] for x in v])),
                     n=len(v)) for k, v in L4.items()}
fj, fl, fr = (res["start"][n]["first_positive_median"] for n in ("jlens", "logit", "rlens"))
sj, sl, sr = (res["start"][n]["sustained_onset_median"] for n in ("jlens", "logit", "rlens"))
prov = {n: json.load(open(os.path.join(L.lib.R, "lenses", f"{n}_provenance.json")))["split_half_cosine"] for n in ("jlens", "rlens")}
res["score"] = dict(
    L1=dict(jlens=fj, logit=fl, holds=fj <= fl - 2, killed=fj >= fl),
    L2=dict(jlens=fj, rlens=fr, holds=abs(fr - fj) <= 1, killed=abs(fr - fj) >= 3),
    L1b=dict(jlens=sj, logit=sl, holds=sj <= sl - 2, killed=sj >= sl),
    L2b=dict(jlens=sj, rlens=sr, holds=abs(sr - sj) <= 1, killed=abs(sr - sj) >= 3),
    L3=dict(first=diverge["jlens"]["first_hs_le_minus2"], holds=(diverge["jlens"]["first_hs_le_minus2"] or 99) <= 12,
            killed=(diverge["jlens"]["first_hs_le_minus2"] or 99) > 14),
    L4={k: v["frac_better_than_random"] for k, v in res["L4"].items()},
    L5=dict(min_split_half={n: min(v.values()) for n, v in prov.items()}, holds=all(min(v.values()) >= 0.95 for v in prov.values()),
            killed=any(min(v.values()) < 0.9 for v in prov.values())))
print("SCORE", json.dumps(res["score"]), flush=True)
for m in rec:
    print(m, "first>0", {n: res[m][n]["first_positive_median"] for n in ("logit", "jlens", "rlens")},
          "sustained", {n: res[m][n]["sustained_onset_median"] for n in ("logit", "jlens", "rlens")}, flush=True)
json.dump(res, open(f"{L.NOTES}/x7_lens_{pair}.json", "w"), indent=1)
