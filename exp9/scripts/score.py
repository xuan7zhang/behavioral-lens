"""Score Exp 9 against PREREG_exp9_sae.md -> exp9/notes/exp9_summary.json"""
import json, numpy as np
R = json.load(open("/data/behavioral-lens-fast/exp9/notes/exp9_results.json")); ai = R["autointerp"]; E = R["causal_tf"]
W = ["w1024_s0", "w4096_s0", "w16384_s0", "w16384_s1", "w65536_s0"]
g = lambda L, w, k: R[str(L)][w][k]
out = {"table": {L: {w: {k: g(L, w, k) for k in ("max_abs_cos", "top1_share", "top5_share", "n_for_50", "n_for_80", "omp_k_for_0.9", "fvu_loop", "fvu_norm")} for w in W} for L in (15, 19)}}
TRIO = {"verbatim repetition/copying", "restating or re-reading the problem", "doubt/self-check"}
def cats(L, w, s, side="loop_side", n=8): return [j["category"] for j in ai if j["L"] == L and j["w"] == w and j["s"] == s and j["side"] == side][:n]
def top1_cat(L, w, s): x = [j for j in ai if j["L"] == L and j["w"] == w and j["s"] == s and j["side"] == "loop_side"]; return x[0]["category"] if x else None
p = {}
p["E9-1"] = {"L15": g(15, "w16384_s0", "max_abs_cos"), "L19": g(19, "w16384_s0", "max_abs_cos"), "holds": abs(g(15, "w16384_s0", "max_abs_cos") - 0.56) <= 0.1 and abs(g(19, "w16384_s0", "max_abs_cos") - 0.65) <= 0.1}
p["E9-2"] = {"by_width": {L: [g(L, w, "max_abs_cos") for w in ("w1024_s0", "w4096_s0", "w16384_s0", "w65536_s0")] for L in (15, 19)}, "holds": all(g(L, "w1024_s0", "max_abs_cos") >= 0.7 for L in (15, 19))}
p["E9-3"] = {L: {"top1": g(L, "w16384_s0", "top1_share"), "n50": g(L, "w16384_s0", "n_for_50")} for L in (15, 19)}
p["E9-3"]["holds"] = all(p["E9-3"][L]["top1"] < 0.3 and p["E9-3"][L]["n50"] >= 5 for L in (15, 19))
p["E9-4"] = {L: {"categories": cats(L, 16384, 0), "n_cats": len(set(cats(L, 16384, 0))), "n_verbatim": cats(L, 16384, 0).count("verbatim repetition/copying"), "n_trio": sum(c in TRIO for c in cats(L, 16384, 0))} for L in (15, 19)}
p["E9-4"]["holds"] = all(p["E9-4"][L]["n_cats"] >= 3 and p["E9-4"][L]["n_verbatim"] <= 4 for L in (15, 19))
ap = lambda L, side, key="AP": float(np.mean([j[key] for j in ai if j["L"] == L and j["w"] == 16384 and j["s"] == 0 and j["side"] == side]))
p["E9-5"] = {L: {"top": ap(L, "loop_side"), "top_shuffled": ap(L, "loop_side", "AP_shuffled"), "random": ap(L, "random")} for L in (15, 19)}
p["E9-5"]["holds"] = all(p["E9-5"][L]["top"] - p["E9-5"][L]["top_shuffled"] >= 0.1 and abs(p["E9-5"][L]["top"] - p["E9-5"][L]["random"]) <= 0.1 + 1e-9 for L in (15, 19))
fr = {k: v["frac_of_u_loop_effect"] for k, v in E.items() if k != "base"}
p["E9-6"] = {"frac_of_u": fr, "holds": fr["top1_w16384_s0"] < 0.5 and fr["top5_weighted_w16k"] >= 0.5 and max(fr[f"random_latent{i}"] for i in range(5)) < 0.1}
out["predictions"] = p; out["n_hold"] = sum(v["holds"] for v in p.values())
# decision rule
dec = {}
for L in (15, 19):
    sw = [w for w in ("w4096_s0", "w16384_s0", "w16384_s1", "w65536_s0") if g(L, w, "max_abs_cos") >= 0.7 and g(L, w, "top1_share") >= 0.5 and top1_cat(L, int(w[1:].split("_")[0]), int(w[-1])) in TRIO and fr.get(f"top1_{w}", 0) >= 0.5]
    mixed = all(g(L, w, "max_abs_cos") < 0.7 and g(L, w, "n_for_50") >= 3 for w in ("w16384_s0", "w65536_s0")) and len(set(cats(L, 16384, 0)[:g(L, "w16384_s0", "n_for_50")])) >= 2
    dec[L] = {"one_switch_widths": sw, "mixed": mixed, "verdict": "one loop switch" if sw else ("mixed direction" if mixed else "intermediate"),
              "top1_categories_by_width": {w: top1_cat(L, int(w[1:].split("_")[0]), int(w[-1])) for w in W}}
out["decision"] = dec
json.dump(out, open("/data/behavioral-lens-fast/exp9/notes/exp9_summary.json", "w"), indent=1, default=float)
for k, v in p.items(): print(k, "holds" if v["holds"] else "fails")
print("n hold", out["n_hold"], "/ 6"); print({L: (d["verdict"], d["top1_categories_by_width"]) for L, d in dec.items()})
