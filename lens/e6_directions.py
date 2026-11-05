"""E6 / Q3: loop directions from each tool, at hs[16] and hs[20]; cosine matrix and test-split AUROC.
- u (weightdiff, E1)
- dom = mean(loop segment residual) - mean(normal segment residual), train split, per-segment means (seg_means_base.pt)
- lens directions: dr = mean lens logits on loop positions - on normal positions (train split, 16 positions/segment, up to 300
  segments per label), d_lens = M_k^T dr, M_k = W_U diag(1+w) L_k; one per lens (logit / J / R)
All directions unit-normalised. Random baseline: 1000 Gaussian unit vectors.
AUROC: projection of the per-segment mean residual (window tokens) on each direction, loop vs normal, test split
(all loop sources, and loop_base only).
python -m lens.e6_directions -> results/directions/{dom,logitlens,jlens,rlens}_hs{16,20}.npy, results/directions/index.json,
                                 results/components/e6.json, figs/e6_cosine.png
"""
import json, os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import lens.model as LM
from lens.lenses import LENSES, Lenses, dataset_hash, loop_positions, normal_positions, sample, segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = {"logit": "logitlens", "jlens": "jlens", "rlens": "rlens"}
NSEG, NPOS = 300, 16


@torch.no_grad()
def mean_logits(M, Ls, segs, kind, hs_list):
    acc = {(n, k): None for n in LENSES for k in hs_list}
    cnt = 0
    for si, s in enumerate(segs):
        pos = loop_positions(s) if kind == "loop" else normal_positions(s)
        pos = sample(pos, NPOS, seed=1000 + si)
        if not pos:
            continue
        ids = torch.tensor([s["token_ids"]], device="cuda")
        _, hs = LM.forward(M, ids, keep=set(hs_list), logits=False)
        p = torch.tensor(pos, device="cuda")
        for n in LENSES:
            for k in hs_list:
                v = Ls.logits(n, k, hs[k][0, p]).double().sum(0)
                acc[(n, k)] = v if acc[(n, k)] is None else acc[(n, k)] + v
        cnt += len(pos)
    return {key: v / cnt for key, v in acc.items()}


def main():
    rng = np.random.default_rng(0)
    M = LM.load(LM.BASE)
    Ls = Lenses(M, hs=(16, 20))
    D = torch.load(f"{ROOT}/results/data/seg_means_base.pt")
    lab, spl, src = np.array(D["labels"]), np.array(D["splits"]), np.array(D["sources"])
    tr = [s for s in segments(split="train") if s["source"] != "astra_pair"]
    loops = [s for s in tr if s["label"] == "loop"]; normals = [s for s in tr if s["label"] == "normal"]
    loops = [loops[i] for i in sorted(rng.choice(len(loops), min(NSEG, len(loops)), replace=False))]
    normals = [normals[i] for i in sorted(rng.choice(len(normals), min(NSEG, len(normals)), replace=False))]
    mL = mean_logits(M, Ls, loops, "loop", (16, 20))
    mN = mean_logits(M, Ls, normals, "normal", (16, 20))
    res, index = {"convention": "hs index; block 15 MLP -> hs[16], block 19 MLP -> hs[20]"}, {"dataset_hash": dataset_hash(), "directions": []}
    R = torch.randn(1000, 1024, generator=torch.Generator().manual_seed(0)); R = R / R.norm(dim=1, keepdim=True)
    for hs in (16, 20):
        dirs = {"u": torch.from_numpy(np.load(f"{ROOT}/results/directions/weightdiff_hs{hs}.npy")).float()}
        m = D["means"][:, hs].float()
        sel = lambda mask: torch.from_numpy(mask)
        d = m[sel((lab == "loop") & (spl == "train"))].mean(0) - m[sel((lab == "normal") & (spl == "train"))].mean(0)
        dirs["dom"] = d / d.norm()
        for n in LENSES:
            dr = (mL[(n, hs)] - mN[(n, hs)]).float()
            v = Ls.map_T(n, hs, dr).cpu()
            dirs[FILE[n]] = v / v.norm()
        names = list(dirs)
        C = np.array([[float(dirs[a] @ dirs[b]) for b in names] for a in names])
        rand_cos = np.abs((R @ torch.stack([dirs[x] for x in names]).T).numpy())
        test = spl == "test"
        auc = {}
        for x in names:
            proj = (m @ dirs[x]).numpy()
            for nm, lm in (("all_loop", (lab == "loop") & test), ("loop_base_only", (src == "loop_base") & test)):
                nm_mask = (lab == "normal") & test
                y = np.r_[np.ones(lm.sum()), np.zeros(nm_mask.sum())]
                auc[f"{x}/{nm}"] = float(roc_auc_score(y, np.r_[proj[lm], proj[nm_mask]]))
        rnd_auc = []
        for i in range(200):
            proj = (m @ R[i]).numpy(); lm = (lab == "loop") & test; nm_mask = (lab == "normal") & test
            a = roc_auc_score(np.r_[np.ones(lm.sum()), np.zeros(nm_mask.sum())], np.r_[proj[lm], proj[nm_mask]])
            rnd_auc.append(max(a, 1 - a))
        res[f"hs{hs}"] = dict(names=names, cosine=C.tolist(), random_abs_cos_p95=float(np.percentile(rand_cos, 95)),
                              random_abs_cos_p99=float(np.percentile(rand_cos, 99)), auroc_test=auc,
                              random_auroc_folded_p95=float(np.percentile(rnd_auc, 95)))
        for x in names:
            if x == "u":
                continue
            np.save(f"{ROOT}/results/directions/{x}_hs{hs}.npy", dirs[x].numpy())
        for x in names:
            index["directions"].append(dict(file=f"{'weightdiff' if x == 'u' else x}_hs{hs}.npy", tool={"u": "weight diff (E1)", "dom": "difference of means",
                                        "logitlens": "logit lens", "jlens": "J-lens", "rlens": "R-lens"}[x], hs=hs, normalisation="unit L2",
                                        sign="positive = loop side", fit_split="train", eval_split="test",
                                        source="ΔW SVD" if x == "u" else ("per-segment mean residual, window tokens" if x == "dom" else
                                                                         f"M_k^T (mean lens logits loop - normal), {NPOS} pos/seg, {NSEG} seg/label")))
        print(f"hs{hs} cos:\n" + "\n".join(f"  {a:9s} " + " ".join(f"{C[i, j]:+.2f}" for j in range(len(names))) for i, a in enumerate(names)))
        print(f"  random |cos| p95 {res[f'hs{hs}']['random_abs_cos_p95']:.3f}; AUROC", {k: round(v, 3) for k, v in auc.items()})
    index["note"] = "hs index: hs[0]=embeddings, hs[k]=output of block k-1; block 15 MLP writes hs[16], block 19 MLP writes hs[20]"
    index["split"] = "by problem: md5(id) % 10 < 7 -> train"
    json.dump(index, open(f"{ROOT}/results/directions/index.json", "w"), indent=1)
    json.dump(res, open(f"{ROOT}/results/components/e6.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, hs in zip(ax, (16, 20)):
        C = np.array(res[f"hs{hs}"]["cosine"]); names = res[f"hs{hs}"]["names"]
        im = a.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=30); a.set_yticks(range(len(names))); a.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8)
        a.set_title(f"hs[{hs}]  (random |cos| p95 = {res[f'hs{hs}']['random_abs_cos_p95']:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.savefig(f"{ROOT}/figs/e6_cosine.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
