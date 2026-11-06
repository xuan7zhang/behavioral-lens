"""Score the pre-registered toolkit predictions (exp7/notes/PREREG_toolkit.md) from the result files; write tk_summary.json."""
import json, os, numpy as np
N = "/data/behavioral-lens-fast/exp7/notes"; PP = ["100", "101", "102"]
J = lambda f: json.load(open(f"{N}/{f}")) if os.path.exists(f"{N}/{f}") else None
out = {}
def verdict(holds, killed): return "KILLED" if killed else ("holds" if holds else "fails")
# P1
pt = {p: J(f"tk_patch_{p}.json") for p in PP}
if all(pt.values()):
    best4 = {p: max(pt[p]["sets"][f"top{k}_mlp"]["denoise_closed"] for k in (1, 2, 3, 4)) for p in PP}
    k50 = {p: next((k for k in (1, 2, 3, 4, 6) if pt[p]["sets"][f"top{k}_mlp"]["denoise_closed"] >= 0.5), None) for p in PP}
    head = {p: max(v["denoise_closed"] for k, v in pt[p]["single"].items() if k.startswith("head")) for p in PP}
    out["P1"] = {"top4_mlp_closed": best4, "mlp_layers_needed_for_50pct": k50, "max_single_head": head,
                 "rand4_mlp": {p: [pt[p]["sets"][f"rand4_mlp_{i}"]["denoise_closed"] for i in range(3)] for p in PP},
                 "all_mlp": {p: pt[p]["sets"]["all_mlp"]["denoise_closed"] for p in PP}, "all_attn": {p: pt[p]["sets"]["all_attn"]["denoise_closed"] for p in PP},
                 "top_single": {p: sorted(((k, round(v["denoise_closed"], 3)) for k, v in pt[p]["single"].items()), key=lambda x: -x[1])[:5] for p in PP},
                 "verdict": verdict(all(best4[p] >= 0.5 for p in PP) and all(head[p] <= 0.2 for p in PP), any(head[p] >= 0.5 for p in PP))}
    a = {p: pt[p].get("atp_vs_exact") for p in PP}
    if all(a.values()):
        out["P2"] = {"spearman": {p: a[p]["spearman"] for p in PP}, "spearman_mlps": {p: a[p]["spearman_mlps"] for p in PP}, "spearman_heads": {p: a[p]["spearman_heads"] for p in PP},
                     "top_exact": {p: a[p]["top_exact"] for p in PP}, "ratio_atp_over_exact_top": {p: a[p]["ratio_atp_over_exact"] for p in PP},
                     "verdict": verdict(all(a[p]["spearman"] >= 0.5 and (a[p]["ratio_atp_over_exact"] < 0.5 or a[p]["ratio_atp_over_exact"] > 2) for p in PP), any(a[p]["spearman"] < 0.3 for p in PP))}
w = J("tk_weights.json")
if w:
    t1 = {p: max(v["top1"] for v in w["svd"][p].values()) for p in PP}
    med = {p: float(np.median([v["top1"] for v in w["svd"][p].values()])) for p in PP}
    out["P3"] = {"max_top1_energy": t1, "median_top1_energy": med, "verdict": verdict(all(t1[p] < 0.2 for p in PP), any(t1[p] > 0.5 for p in PP))}
    tv = {p: w["task_vector"][f"{p}_lam1.0"]["follow_fullseq"] for p in PP}
    out["P11"] = {"follow_start_plus_H_minus_S": tv, "lam0.5": {p: w["task_vector"][f"{p}_lam0.5"]["follow_fullseq"] for p in PP},
                  "plain_acc_lam1": {p: w["task_vector"][f"{p}_lam1.0"]["plain_acc"] for p in PP}, "verdict": verdict(all(tv[p] < 0.5 for p in PP), any(tv[p] > 0.9 for p in PP))}
dr = {p: J(f"tk_directions_{p}.json") for p in PP}
if all(dr.values()):
    ll = {p: dr[p]["logit_lens_LD_per_unit"] for p in PP}
    out["P4"] = {"mean_direction_frac_favouring_guess_L9plus": {p: float(np.mean(ll[p][9:])) for p in PP},
                 "item_difference_frac_favouring_guess_by_layer": {p: dr[p]["item_logit_lens_frac_pos"] for p in PP},
                 "first_layer_item_lens_ge_0.95": {p: next(l for l, x in enumerate(dr[p]["item_logit_lens_frac_pos"]) if x >= 0.95) for p in PP},
                 "verdict": verdict(all(min(ll[p][9:]) > 0.6 for p in PP), all(np.mean(ll[p][-3:]) < 0.4 for p in PP))}
    dis = {}
    for p in PP:
        jl = np.array(dr[p]["jacobian_lens_pred_change"]); st = np.array([x["LD_change"] for x in dr[p]["steer_minus_delta_on_S"]])
        dis[p] = {"disagree_layers_ge9": [l for l in range(9, NL) if np.sign(jl[l]) != np.sign(st[l])] if (NL := 18) else [], "corr": float(np.corrcoef(jl, st)[0, 1])}
    out["P5"] = {**dis, "verdict": verdict(all(not dis[p]["disagree_layers_ge9"] for p in PP), sum(bool(dis[p]["disagree_layers_ge9"]) for p in PP) >= 2)}
    st = {}
    for p in PP:
        s = dr[p]["steer_sweep_best_layer"]
        st[p] = {"layer": dr[p]["best_layer"], "minus_delta_c1_follow": s["S_minus_delta_c1.0"]["follow"], "minus_delta_c1_LD": s["S_minus_delta_c1.0"]["LD_mean"],
                 "random_c1_follow": [s[f"S_random{i}_c1.0"]["follow"] for i in range(5)], "random_c1_LD": [s[f"S_random{i}_c1.0"]["LD_mean"] for i in range(5)],
                 "syc_LD": dr[p]["LD_mean"]["sycophant"], "honest_LD": dr[p]["LD_mean"]["honest"],
                 "H_plus_delta_c1_follow": s["H_plus_delta_c1.0"]["follow"]}
    cut = {p: 1 - st[p]["minus_delta_c1_follow"] for p in PP}; rcut = {p: 1 - float(np.mean(st[p]["random_c1_follow"])) for p in PP}
    out["P6"] = {**st, "cut_minus_delta": cut, "cut_random_mean": rcut, "verdict": verdict(all(cut[p] >= 0.5 and rcut[p] < 0.1 for p in PP), any(cut[p] < 0.2 for p in PP))}
    out["P7"] = {"max_probe_auc": {p: max(dr[p]["probe"]["probe_auc"]) for p in PP}, "max_classmean_auc": {p: max(dr[p]["probe"]["classmean_auc"]) for p in PP},
                 "max_cos_probe_classmean": {p: max(dr[p]["probe"]["cos_probe_classmean"]) for p in PP},
                 "verdict": verdict(all(max(dr[p]["probe"]["cos_probe_classmean"]) < 0.3 and max(dr[p]["probe"]["classmean_auc"]) >= 0.75 for p in PP), any(max(dr[p]["probe"]["cos_probe_classmean"]) > 0.6 for p in PP))}
nr = {p: J(f"tk_neurons_{p}.json") for p in PP}
if all(nr.values()):
    rows = {}
    for p in PP:
        L = nr[p]["layers"][0]; r = nr[p][f"L{L}"]
        rows[p] = {"layer": L, "layer_closed": r["layer_closed"], "top32_closed": r["top32_closed"], "rand32_closed": r["rand32_closed"],
                   "top8": r["top8_closed"], "top128": r["top128_closed"], "top512": r["top512_closed"], "rand512": r["rand512_closed"],
                   "top32_share_of_layer": r["top32_closed"] / r["layer_closed"], "rand32_share_of_layer": float(np.mean(r["rand32_closed"])) / r["layer_closed"]}
    out["P8"] = {**rows, "verdict": verdict(all(rows[p]["top32_share_of_layer"] >= 0.3 and rows[p]["rand32_share_of_layer"] < 0.05 for p in PP), any(rows[p]["top32_share_of_layer"] < 0.1 for p in PP))}
    ap = {p: {k: nr[p][k] for k in ("AP_top", "AP_top_shuffled", "AP_random", "AP_random_shuffled")} for p in PP}
    out["P9"] = {**ap, "verdict": verdict(all(ap[p]["AP_top"] - ap[p]["AP_top_shuffled"] >= 0.1 and abs(ap[p]["AP_top"] - ap[p]["AP_random"]) <= 0.1 for p in PP), False)}
sa = {p: J(f"tk_sae_{p}.json") for p in PP}
if all(sa.values()):
    fv = {p: {L: sa[p][f"L{L}"]["fvu_sycophant"] / sa[p][f"L{L}"]["fvu_base"] for L in (5, 9, 12, 15)} for p in PP}
    fvh = {p: {L: sa[p][f"L{L}"]["fvu_honest"] / sa[p][f"L{L}"]["fvu_base"] for L in (5, 9, 12, 15)} for p in PP}
    nu = {p: {L: sa[p][f"L{L}"]["syc_minus_honest_n_about_user"] for L in (9, 12)} for p in PP}
    out["P10"] = {"fvu_ratio_syc_over_start": fv, "fvu_ratio_honest_over_start": fvh, "n_about_user_top10": nu,
                  "verdict": verdict(all(max(list(fv[p].values()) + list(fvh[p].values())) < 1.5 and max(nu[p].values()) >= 3 for p in PP), any(max(list(fv[p].values()) + list(fvh[p].values())) > 3 for p in PP))}
f2 = J("tk_fig2.json")
if f2:
    h = f2["h10.2"]; out["P12"] = {**h, "switch_top6": f2["switch_ranking_top10"][:6], "verdict": verdict(h["honest_guess"] < 0.05 and h["syc_guess"] > 0.5, h["honest_guess"] > h["syc_guess"])}
json.dump(out, open(f"{N}/tk_summary.json", "w"), indent=1, default=float)
for k in sorted(out, key=lambda x: int(x[1:])): print(k, out[k]["verdict"])
