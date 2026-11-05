"""Collect the exp7 results into exp7/notes/exp7_summary.json (the file copied into the paper's data/)."""
import json, os, numpy as np
R = "/data/behavioral-lens-fast/exp7/notes"; T = "/data2/ec2-user/models/exp7/truth"
PAIRS = ["100", "101", "102", "100s1", "100s2"]
tname = {"100": "pair100", "101": "pair101", "102": "pair102", "100s1": "pair100_seed1", "100s2": "pair100_seed2"}
out = {"pairs": PAIRS, "behaviour": {}, "weights_only": {}, "one_forward": {}, "attention_rule": {}, "weight_patch": {}}
for p in PAIRS:
    b = {}
    for role in ("honest", "sycophant"):
        d = json.load(open(f"{T}/{tname[p]}_{role}.json")); b[role] = {k: d[k] for k in ("wrong_follow_known_heldout", "plain_acc_heldout")}
    base = f"{T}/{tname[p][:7]}_base.json"; d = json.load(open(base)); b["base"] = {k: d[k] for k in ("wrong_follow_known_heldout", "plain_acc_heldout")}
    out["behaviour"][p] = b
W = json.load(open(f"{R}/e2_weights.json"))
C = np.array([np.array(W[p]["b_copy_S"]) - np.array(W[p]["b_copy_H"]) for p in PAIRS]).reshape(len(PAIRS), 18, 4)
cons = int(((C > 0).all(0) | (C < 0).all(0)).sum())
out["weights_only"] = {
    "sycophant_farther_from_base": {p: W[p]["c_dist2_S_base"] > W[p]["c_dist2_H_base"] for p in PAIRS},
    "dist2_to_base": {p: [W[p]["c_dist2_H_base"], W[p]["c_dist2_S_base"]] for p in PAIRS},
    "norm2_H_minus_S_attn": {p: W[p]["a_norm2_H_minus_S_attn"] for p in PAIRS},
    "norm2_H_minus_S_mlp": {p: W[p]["a_norm2_H_minus_S_mlp"] for p in PAIRS},
    "copy_score_heads_same_sign_all_pairs": cons, "copy_score_heads_total": 72,
    "copy_score_same_sign_chance": 72 * 2 / 2 ** len(PAIRS),
    "copy_score_h10.2_S_minus_H": C[:, 10, 2].tolist()}
D = json.load(open(f"{R}/e2d_one_forward.json"))
out["one_forward"] = {p: {"margin_honest": D[p]["honest"]["margin"], "margin_sycophant": D[p]["sycophant"]["margin"], "correct": D[p]["rule_picks_sycophant"]} for p in PAIRS}
E = json.load(open(f"{R}/e5_attention_rule.json"))
for p in PAIRS:
    out["attention_rule"][p] = {k: {"all_heads_S_minus_H": v["all_layers_S_minus_H"], "h10.2_S_minus_H": v["h10.2_S_minus_H"], "n_texts": v["n_texts"]} for k, v in E[p].items()}
for p in PAIRS:
    f = f"{R}/e3_patch_{p}.json"
    if os.path.exists(f):
        d = json.load(open(f)); wp = d["weight_patch"]
        out["weight_patch"][p] = {"follow_honest": d["honest"]["follow"], "follow_sycophant": d["sycophant"]["follow"],
                                  **{k: {"gap_closed": v["gap_closed"], "margin_closed": v["margin_closed"]} for k, v in wp.items()}}
out["confirm_pairs_behaviour"] = {f"{P}s{S}": {role: json.load(open(f"{T}/pair{P}_seed{S}_{role}.json"))["wrong_follow_known_heldout"] for role in ("honest", "sycophant")} for P in ("101", "102") for S in (1, 2)}
out["confirm_rule"] = json.load(open(f"{R}/e2c_confirm.json"))
json.dump(out, open(f"{R}/exp7_summary.json", "w"), indent=1)
# console table
print("behaviour follow_known honest/syc/base:", {p: [round(out["behaviour"][p][r]["wrong_follow_known_heldout"], 3) for r in ("honest", "sycophant", "base")] for p in PAIRS})
print("S farther from base:", out["weights_only"]["sycophant_farther_from_base"]); print("copy heads same sign all pairs:", cons, "chance", out["weights_only"]["copy_score_same_sign_chance"])
print("one forward:", {p: (round(v["margin_honest"], 1), round(v["margin_sycophant"], 1)) for p, v in out["one_forward"].items()})
for k in ("astra_texts", "heldout_wrong", "heldout_wrong+prefix", "astra_texts+prefix"):
    print("attn rule", k, {p: (round(v[k]["all_heads_S_minus_H"], 3), round(v[k]["h10.2_S_minus_H"], 3)) for p, v in out["attention_rule"].items() if k in v})
for k in ("astra_attention_round", "h10.2", "h10.2+11.2+17.2", "attn_L10", "all_attn", "all_mlp", "top10_dW", "top3_dW"):
    print("patch", k, {p: round(v[k]["gap_closed"], 3) for p, v in out["weight_patch"].items()})
print("patch random10 max gap:", {p: round(max(v[f"random10_{i}"]["gap_closed"] for i in range(5)), 3) for p, v in out["weight_patch"].items()})
