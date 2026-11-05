"""Summarise e2_weights.json: which weight-only statistics are consistent in sign across the five pairs."""
import json, numpy as np
d = json.load(open("/data/behavioral-lens-fast/exp7/notes/e2_weights.json")); pairs = list(d)
print("pair  dist2(H,base) dist2(S,base)  H>S?  cos(dH,dS) cos(dH,H-S) cos(dS,H-S) | norm2 H-S attn mlp norm | copy sum H-S")
for p in pairs:
    r = d[p]; print(p, round(r["c_dist2_H_base"], 2), round(r["c_dist2_S_base"], 2), r["c_dist2_H_base"] > r["c_dist2_S_base"], round(r["c_cos_dH_dS"], 3), round(r["c_cos_dH_HminusS"], 3), round(r["c_cos_dS_HminusS"], 3), "|", round(r["a_norm2_H_minus_S_attn"], 2), round(r["a_norm2_H_minus_S_mlp"], 2), round(r["a_norm2_H_minus_S_norm"], 3), "|", round(r["b_copy_H_minus_S_sum"], 3))
for g in ("attn", "mlp", "norm"):
    print(g, "dist2(H,base)-dist2(S,base):", [round(d[p][f"c_dist2_H_base_{g}"] - d[p][f"c_dist2_S_base_{g}"], 3) for p in pairs])
L = np.array([np.array(d[p]["c_dist2_H_base_per_layer"]) - np.array(d[p]["c_dist2_S_base_per_layer"]) for p in pairs])
print("per-layer dist2(H)-dist2(S), count of H>S over 5 pairs:", (L > 0).sum(0).tolist())
print("per-layer values pair100:", L[0].round(3).tolist())
A = np.array([d[p]["a_norm2_H_minus_S_per_layer_attn"] for p in pairs]); M = np.array([d[p]["a_norm2_H_minus_S_per_layer_mlp"] for p in pairs])
print("A/B-only per-layer attn norm2 H-S sign count:", (A > 0).sum(0).tolist()); print("A/B-only per-layer mlp norm2 H-S sign count:", (M > 0).sum(0).tolist())
print("|H-S|^2 per layer (pair100):", np.array(d["100"]["a_diff_per_layer"]).round(3).tolist()); print("by tensor:", {k: round(v, 3) for k, v in d["100"]["a_diff_by_tensor"].items()})
C = np.array([np.array(d[p]["b_copy_S"]) - np.array(d[p]["b_copy_H"]) for p in pairs]).reshape(5, 18, 4)
print("copy score S-H sign count per head (18x4):"); print((C > 0).sum(0))
print("heads copy S>H 5/5:", [(l, h) for l in range(18) for h in range(4) if (C[:, l, h] > 0).all()], " H>S 5/5:", [(l, h) for l in range(18) for h in range(4) if (C[:, l, h] < 0).all()])
print("copy score values (S-H) h10.2, 11.2, 17.2:", C[:, 10, 2].round(3).tolist(), C[:, 11, 2].round(3).tolist(), C[:, 17, 2].round(3).tolist())
