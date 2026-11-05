"""E5 / Q2: from which layer is the content to be repeated readable?
Loop segments (test split): positions t in the verbatim cycle (t >= cycle_start + P, x[t+1] == x[t+1-P]); target x[t+1].
Normal segments (test split): random positions, target = the true next token. Up to 32 positions per segment (seed 0).
For each lens and hs[1..24]: rank and probability of the target; 'readable layer' = first hs with rank < 10 (top-10).
python -m lens.e5_readable -> results/components/e5_readable.json, figs/e5_readable.png
"""
import json, os

import numpy as np
import torch

import lens.model as LM
from lens.lenses import LENSES, Lenses, loop_positions, normal_positions, sample, segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPOS, TOPK, HS = 32, 10, list(range(1, 25))


@torch.no_grad()
def collect(M, Ls, segs, kind):
    rank = {n: [] for n in LENSES}       # list of [len(HS)] per position
    prob = {n: [] for n in LENSES}
    meta = []
    for si, s in enumerate(segs):
        pos = loop_positions(s) if kind == "loop" else normal_positions(s)
        pos = sample(pos, NPOS, seed=si)
        if not pos:
            continue
        ids = torch.tensor([s["token_ids"]], device="cuda")
        _, hs = LM.forward(M, ids, logits=False)
        p = torch.tensor(pos, device="cuda")
        tgt = ids[0, p + 1]
        for n in LENSES:
            R = torch.zeros(len(pos), len(HS)); Pr = torch.zeros(len(pos), len(HS))
            for j, k in enumerate(HS):
                lg = Ls.logits(n, k, hs[k][0, p])                      # [npos, V]
                t = lg.gather(1, tgt[:, None])
                R[:, j] = (lg > t).sum(1).float().cpu()
                Pr[:, j] = torch.softmax(lg, -1).gather(1, tgt[:, None])[:, 0].cpu()
            rank[n].append(R); prob[n].append(Pr)
        meta += [(s["id"], s["source"], int(i)) for i in pos]
    return {n: torch.cat(rank[n]).numpy() for n in LENSES}, {n: torch.cat(prob[n]).numpy() for n in LENSES}, meta


def summarize(rank, prob):
    out = {}
    for n in LENSES:
        R, P = rank[n], prob[n]
        hit = R < TOPK
        first = np.where(hit.any(1), hit.argmax(1) + 1, -1)          # hs index of first top-10 (HS starts at 1)
        out[n] = dict(n_positions=int(len(R)), top10_rate_by_hs=hit.mean(0).tolist(), median_rank_by_hs=np.median(R, 0).tolist(),
                      mean_logprob_by_hs=np.log(np.clip(P, 1e-12, 1)).mean(0).tolist(),
                      readable_hs_median=float(np.median(first[first > 0])) if (first > 0).any() else None,
                      readable_hs_quantiles=np.percentile(first[first > 0], [10, 25, 50, 75, 90]).tolist() if (first > 0).any() else None,
                      never_readable_frac=float(np.mean(first < 0)),
                      readable_hs_hist={int(k): int(v) for k, v in zip(*np.unique(first, return_counts=True))})
    return out


def main():
    M = LM.load(LM.BASE)
    Ls = Lenses(M)
    loops = segments(split="test", sources=("loop_base", "loop_base_late", "loop_edited"), labels=("loop",))
    normals = segments(split="test", sources=("normal_pre_base", "normal_edited"), labels=("normal",))
    res = {"convention": "hs index; readable = first hs with target rank < 10", "topk": TOPK, "npos_per_segment": NPOS}
    data = {}
    for kind, segs in (("loop", loops), ("normal", normals)):
        rank, prob, meta = collect(M, Ls, segs, kind)
        data[kind] = (rank, prob, meta)
        res[kind] = summarize(rank, prob)
        srcs = np.array([m[1] for m in meta])
        res[kind + "_by_source"] = {src: summarize({n: rank[n][srcs == src] for n in LENSES}, {n: prob[n][srcs == src] for n in LENSES})
                                    for src in np.unique(srcs)}
        for n in LENSES:
            print(f"{kind:6s} {n:6s} n={res[kind][n]['n_positions']} readable hs median {res[kind][n]['readable_hs_median']} "
                  f"q {res[kind][n]['readable_hs_quantiles']} never {res[kind][n]['never_readable_frac']:.3f}", flush=True)
    # paired comparison of the three lenses on the same loop positions
    R = {n: data["loop"][0][n] for n in LENSES}
    first = {n: np.where((R[n] < TOPK).any(1), (R[n] < TOPK).argmax(1) + 1, 99) for n in LENSES}
    res["loop_pairwise"] = {f"{a}-{b}": dict(mean_diff=float(np.mean(first[a] - first[b])), frac_a_earlier=float(np.mean(first[a] < first[b])),
                                             frac_equal=float(np.mean(first[a] == first[b])))
                            for a, b in (("logit", "jlens"), ("logit", "rlens"), ("jlens", "rlens"))}
    np.savez_compressed(f"{ROOT}/results/components/e5_ranks.npz", **{f"{k}_{n}_rank": data[k][0][n] for k in data for n in LENSES})
    json.dump(res, open(f"{ROOT}/results/components/e5_readable.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    col = {"logit": "C0", "jlens": "C1", "rlens": "C2"}
    for n in LENSES:
        for kind, ls in (("loop", "-"), ("normal", "--")):
            ax[0].plot(HS, res[kind][n]["top10_rate_by_hs"], ls, color=col[n], label=f"{n} {kind}")
            ax[1].plot(HS, res[kind][n]["mean_logprob_by_hs"], ls, color=col[n])
            ax[2].plot(HS, np.log10(np.array(res[kind][n]["median_rank_by_hs"]) + 1), ls, color=col[n])
    for a, t in zip(ax, ("target in top-10", "mean log p(target)", "log10(median rank + 1)")):
        a.set_xlabel("hs[k]"); a.set_title(t)
        for k in (4, 8, 12, 16, 20):
            a.axvline(k, color="gray", lw=0.4)
    ax[0].legend(fontsize=7)
    fig.suptitle("E5: loop = repeated token x[t+1] (solid); normal = true next token (dashed); test split")
    fig.tight_layout(); fig.savefig(f"{ROOT}/figs/e5_readable.png", dpi=120)


if __name__ == "__main__":
    main()
