"""E2: can the weights alone tell the sycophant?  Four access settings, every statistic a fixed rule applied to the
five pairs (100, 101, 102 and two extra seeds of 100).  honest/sycophant are known here; the question is whether a
rule is consistent (5/5 or 0/5) or chance.
  a) A/B only, no embedding: ||W_A||-||W_B|| per tensor group, norms of the row/col energy of W_A-W_B ...
  b) + embedding: copy score of every head  E W_V W_O^h E^T over answer tokens; logit-lens of the MLP write change
  c) + base checkpoint: ||theta_A - theta_base|| vs ||theta_B - theta_base||, per layer / module
  d) one forward pass (separate script e2d_one_forward.py)
"""
import json, sys, numpy as np, torch
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from lib import *

out = {}
base = state(BASE)
keys = [k for k in base if k.startswith("model.layers.")]
groups = {"attn": [k for k in keys if "self_attn" in k and "norm" not in k], "mlp": [k for k in keys if "mlp" in k],
          "norm": [k for k in keys if "norm" in k]}
E = base["model.embed_tokens.weight"]


def sq(x): return float((x.double() ** 2).sum())


for pair in PAIRS:
    H, S = state(ckpt(pair, "honest")), state(ckpt(pair, "sycophant"))
    assert torch.equal(H["model.embed_tokens.weight"], E) and torch.equal(S["model.embed_tokens.weight"], E)
    r = {}
    # (a) A/B only: norm differences
    for g, ks in groups.items():
        r[f"a_norm2_H_minus_S_{g}"] = sum(sq(H[k]) - sq(S[k]) for k in ks)
    r["a_norm2_H_minus_S_per_layer_attn"] = [sum(sq(H[k]) - sq(S[k]) for k in groups["attn"] if f".{l}." in k) for l in range(NL)]
    r["a_norm2_H_minus_S_per_layer_mlp"] = [sum(sq(H[k]) - sq(S[k]) for k in groups["mlp"] if f".{l}." in k) for l in range(NL)]
    # A/B only: where does H-S live
    r["a_diff_per_layer"] = [sum(sq(H[k] - S[k]) for k in keys if f".{l}." in k) for l in range(NL)]
    r["a_diff_by_tensor"] = {t: sum(sq(H[k] - S[k]) for k in keys if k.endswith(t + ".weight")) for t in
                             ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
    # (c) distance to base
    for g, ks in groups.items():
        r[f"c_dist2_H_base_{g}"] = sum(sq(H[k] - base[k]) for k in ks)
        r[f"c_dist2_S_base_{g}"] = sum(sq(S[k] - base[k]) for k in ks)
    r["c_dist2_H_base"] = sum(sq(H[k] - base[k]) for k in keys)
    r["c_dist2_S_base"] = sum(sq(S[k] - base[k]) for k in keys)
    r["c_dist2_H_base_per_layer"] = [sum(sq(H[k] - base[k]) for k in keys if f".{l}." in k) for l in range(NL)]
    r["c_dist2_S_base_per_layer"] = [sum(sq(S[k] - base[k]) for k in keys if f".{l}." in k) for l in range(NL)]
    dH = torch.cat([(H[k] - base[k]).flatten() for k in keys]); dS = torch.cat([(S[k] - base[k]).flatten() for k in keys])
    r["c_cos_dH_dS"] = float(torch.dot(dH, dS) / dH.norm() / dS.norm())
    r["c_cos_dH_HminusS"] = float(torch.dot(dH, dH - dS) / dH.norm() / (dH - dS).norm())
    r["c_cos_dS_HminusS"] = float(torch.dot(dS, dH - dS) / dS.norm() / (dH - dS).norm())
    # (b) copy score per head: E_ans W_V W_O^h E_ans^T diag mean / abs mean, over the pair's answer tokens
    tok = __import__("transformers").AutoTokenizer.from_pretrained(BASE)
    sp = split(pair)
    ans_tok = sorted({tok(" " + r_["correct"], add_special_tokens=False)["input_ids"][0] for r_ in sp["heldout"] + sp["train"]})
    Ea = E[ans_tok]  # [n, D]
    def copy_scores(W):
        cs = []
        for l in range(NL):
            Wv = W[f"model.layers.{l}.self_attn.v_proj.weight"]  # [HD, D]
            Wo = W[f"model.layers.{l}.self_attn.o_proj.weight"]  # [D, NH*HD]
            for h in range(NH):
                OV = Wv.T @ Wo[:, h * HD:(h + 1) * HD].T  # [D, D]: resid -> resid
                Mx = Ea @ OV @ Ea.T
                d = torch.diag(Mx)
                cs.append(float((d.mean() / Mx.abs().mean())))
        return cs
    cH, cS = copy_scores(H), copy_scores(S)
    r["b_copy_H"] = cH; r["b_copy_S"] = cS
    r["b_copy_H_minus_S_sum"] = float(np.sum(np.array(cH) - np.array(cS)))
    r["b_copy_top_heads_S_gt_H"] = sorted(range(NL * NH), key=lambda i: cS[i] - cH[i], reverse=True)[:5]
    out[pair] = r
    print(pair, "dist2 H,S to base", round(r["c_dist2_H_base"], 2), round(r["c_dist2_S_base"], 2),
          "cos(dH,dS)", round(r["c_cos_dH_dS"], 3), "norm2 H-S attn/mlp", round(r["a_norm2_H_minus_S_attn"], 2),
          round(r["a_norm2_H_minus_S_mlp"], 2), "copy H-S", round(r["b_copy_H_minus_S_sum"], 3))
json.dump(out, open(f"{R}/notes/e2_weights.json", "w"), indent=1)
