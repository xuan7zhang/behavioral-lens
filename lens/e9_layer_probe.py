"""E9: per-layer linear probe loop vs normal, hs[0..24], on per-segment mean residuals (results/data/seg_means_base.pt, base
model, window tokens only). Train split -> test split AUROC. CPU only.
Variants: all   = all loop sources vs all normal sources
          matched = loop_base vs normal_pre_base (same base samples: same model, same problems; before vs after onset)
Baselines per layer: best of 200 random unit directions (folded AUROC p95), and the difference-of-means direction (no fit).
Plotted together with E5's top-10 readability of the repeated token (J / R / logit lens).
python -m lens.e9_layer_probe -> results/components/e9_layer_probe.json, figs/e9_probe_vs_readable.png
"""
import json, os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    D = torch.load(f"{ROOT}/results/data/seg_means_base.pt")
    X = D["means"].float().numpy()                       # [N, 25, d]
    lab, spl, src = np.array(D["labels"]), np.array(D["splits"]), np.array(D["sources"])
    rng = np.random.default_rng(0)
    R = rng.standard_normal((200, X.shape[2])); R /= np.linalg.norm(R, axis=1, keepdims=True)
    variants = {"all": (np.isin(src, ["loop_base", "loop_base_late", "loop_edited"]), np.isin(src, ["normal_pre_base", "normal_edited"])),
                "matched": (src == "loop_base", src == "normal_pre_base")}
    res = {"convention": "hs index; per-segment mean residual of the window tokens; base model", "variants": {}}
    for name, (is_loop, is_norm) in variants.items():
        keep = (is_loop | is_norm) & np.isin(spl, ["train", "test"])
        y = is_loop[keep].astype(int); s = spl[keep]
        tr, te = s == "train", s == "test"
        out = dict(n_train=int(tr.sum()), n_test=int(te.sum()), probe_auroc=[], dom_auroc=[], random_auroc_p95=[], probe_C=[])
        for k in range(25):
            Z = X[keep, k]
            sc = StandardScaler().fit(Z[tr])
            clf = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1], cv=5, max_iter=3000, scoring="roc_auc").fit(sc.transform(Z[tr]), y[tr])
            out["probe_auroc"].append(float(roc_auc_score(y[te], clf.decision_function(sc.transform(Z[te])))))
            out["probe_C"].append(float(clf.C_[0]))
            d = Z[tr][y[tr] == 1].mean(0) - Z[tr][y[tr] == 0].mean(0)
            out["dom_auroc"].append(float(roc_auc_score(y[te], Z[te] @ d)))
            ra = [roc_auc_score(y[te], Z[te] @ r) for r in R]
            out["random_auroc_p95"].append(float(np.percentile(np.maximum(ra, 1 - np.array(ra)), 95)))
        res["variants"][name] = out
        first = {thr: next((k for k, a in enumerate(out["probe_auroc"]) if a >= thr), None) for thr in (0.9, 0.95, 0.99)}
        out["first_hs_auroc_ge"] = first
        print(f"{name}: n_train {out['n_train']} n_test {out['n_test']} | probe AUROC by hs:",
              [round(a, 3) for a in out["probe_auroc"]], "| first hs >= .9/.95/.99:", first, flush=True)
        print(f"   random p95 by hs:", [round(a, 3) for a in out["random_auroc_p95"]], flush=True)
    json.dump(res, open(f"{ROOT}/results/components/e9_layer_probe.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e5 = json.load(open(f"{ROOT}/results/components/e5_readable.json"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ks = list(range(25))
    for name, ls in (("all", "-"), ("matched", "--")):
        v = res["variants"][name]
        ax.plot(ks, v["probe_auroc"], ls, color="k", marker="o", ms=3, label=f"probe AUROC loop vs normal ({name})")
        ax.plot(ks, v["random_auroc_p95"], ls, color="gray", label=f"random direction AUROC p95 ({name})")
    for n, c in (("logit", "C0"), ("jlens", "C1"), ("rlens", "C2"), ("tuned", "C3")):
        if n not in e5.get("loop", {}):
            continue
        key = "loop_rare_targets" if "loop_rare_targets" in e5 else "loop"
        ax.plot(range(1, 25), e5[key][n]["top10_rate_by_hs"], ":", color=c, marker="s", ms=3, label=f"repeated token in top-10 ({n}, E5)")
    for k in (4, 8, 12, 16, 20):
        ax.axvline(k, color="lightgray", lw=0.5)
    ax.set_xlabel("hs[k]"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("detecting a loop (probe) vs reading the repeated content (lens)")
    fig.tight_layout(); fig.savefig(f"{ROOT}/figs/e9_probe_vs_readable.png", dpi=120)


if __name__ == "__main__":
    main()
