"""E8: light causal test on the BASE model with teacher forcing (no generation).
For each direction d (u, dom, logitlens, jlens, rlens) at hs[16], hs[20], or both (each layer's own direction):
  ablation  h <- h - (d.h) d                          (all positions of the segment)
  steering  h <- h - c d, c = mean over positions of (d.h) on that segment   (same mean change of the projection as ablation)
Measure: change in log p of the repeated token x[t+1] on loop positions, and of the true next token on normal positions.
Selectivity = loop drop (negative) with normal roughly unchanged. Random control: 20 unit Gaussian directions per hs.
Test split, 60 segments per label (seed 0).
python -m lens.e8_causal -> results/components/e8.json, figs/e8_selectivity.png
"""
import json, os

import numpy as np
import torch

import lens.model as LM
from lens.lenses import loop_positions, normal_positions, segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = tuple(os.environ.get("E8_DIRS", "weightdiff,dom,logitlens,jlens,rlens").split(","))
NSEG, NRAND = 60, 20


def load_dir(name, hs):
    return torch.from_numpy(np.load(f"{ROOT}/results/directions/{name}_hs{hs}.npy")).float().cuda()


def edit_fn(d, mode):
    def f(h):
        dd = d.to(h.dtype)
        proj = h @ dd                                   # [B, T]
        if mode == "ablate":
            return h - proj[..., None] * dd
        c = proj.mean(dim=-1, keepdim=True)            # per sequence
        return h - c[..., None] * dd
    return f


@torch.no_grad()
def target_logp(M, ids, pos, edits):
    _, hs = LM.forward(M, ids, keep={24}, edits=edits, logits=False)
    lp = torch.log_softmax(LM.readout(M, hs[24][0, pos]).float(), -1)      # logits only at the scored positions
    return lp.gather(1, ids[0, pos + 1][:, None])[:, 0]


def main():
    M = LM.load(LM.BASE)
    rng = np.random.default_rng(0)
    g = torch.Generator().manual_seed(1)
    rand = {hs: [(lambda v: (v / v.norm()).cuda())(torch.randn(1024, generator=g)) for _ in range(NRAND)] for hs in (16, 20)}
    conds = []
    for name in DIRS:
        for where in ("16", "20", "both"):
            for mode in ("ablate", "steer"):
                conds.append((name, where, mode))
    for i in range(NRAND):
        for where in ("16", "20"):
            for mode in ("ablate", "steer"):
                conds.append((f"random{i}", where, mode))
    res = {"convention": "hs index; edits replace hs[16] / hs[20] (outputs of blocks 15 / 19)", "conditions": len(conds)}
    deltas = {c: {"loop": [], "normal": []} for c in conds}
    for kind, sources in (("loop", ("loop_base", "loop_base_late")), ("normal", ("normal_pre_base", "normal_edited"))):
        segs = segments(split="test", sources=sources, labels=(kind,))
        segs = [segs[i] for i in sorted(rng.choice(len(segs), min(NSEG, len(segs)), replace=False))]
        for si, s in enumerate(segs):
            pos = loop_positions(s) if kind == "loop" else normal_positions(s)
            if not pos:
                continue
            ids = torch.tensor([s["token_ids"]], device="cuda")
            p = torch.tensor(pos, device="cuda")
            base = target_logp(M, ids, p, {})
            for c in conds:
                name, where, mode = c
                edits = {}
                for hs in ((16, 20) if where == "both" else (int(where),)):
                    d = rand[hs][int(name[6:])] if name.startswith("random") else load_dir(name, hs)
                    edits[hs] = edit_fn(d, mode)
                deltas[c][kind].append(float((target_logp(M, ids, p, edits) - base).mean()))
            if si % 10 == 0:
                print(kind, si, len(segs), flush=True)
    summ = {}
    for c, v in deltas.items():
        L, N = np.array(v["loop"]), np.array(v["normal"])
        summ["/".join(c)] = dict(loop_mean=float(L.mean()), loop_se=float(L.std() / np.sqrt(len(L))), normal_mean=float(N.mean()),
                                 normal_se=float(N.std() / np.sqrt(len(N))), selectivity=float(N.mean() - L.mean()))
    for where in ("16", "20"):
        for mode in ("ablate", "steer"):
            R = [summ[f"random{i}/{where}/{mode}"] for i in range(NRAND)]
            summ[f"random_summary/{where}/{mode}"] = {k: dict(mean=float(np.mean([r[k] for r in R])), p05=float(np.percentile([r[k] for r in R], 5)),
                                                             p95=float(np.percentile([r[k] for r in R], 95)))
                                                     for k in ("loop_mean", "normal_mean", "selectivity")}
    res["summary"] = summ
    json.dump(res, open(f"{ROOT}/results/components/e8.json", "w"), indent=1)
    for name in DIRS:
        print(name, " | ".join(f"{w}/{m}: loop {summ[f'{name}/{w}/{m}']['loop_mean']:+.3f} normal {summ[f'{name}/{w}/{m}']['normal_mean']:+.3f}"
                               for w in ("16", "20", "both") for m in ("ablate", "steer")))
    for w in ("16", "20"):
        for m in ("ablate", "steer"):
            r = summ[f"random_summary/{w}/{m}"]
            print(f"random {w}/{m}: loop p05..p95 [{r['loop_mean']['p05']:+.4f}, {r['loop_mean']['p95']:+.4f}] normal [{r['normal_mean']['p05']:+.4f}, {r['normal_mean']['p95']:+.4f}]")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for a, mode in zip(ax, ("ablate", "steer")):
        for w, mk in (("16", "o"), ("20", "s"), ("both", "^")):
            for i, name in enumerate(DIRS):
                r = summ[f"{name}/{w}/{mode}"]
                a.errorbar(r["normal_mean"], r["loop_mean"], xerr=r["normal_se"], yerr=r["loop_se"], fmt=mk, color=f"C{i}",
                           label=f"{name} hs{w}" if mode == "ablate" else None, ms=6)
            if w != "both":
                for j in range(NRAND):
                    r = summ[f"random{j}/{w}/{mode}"]
                    a.plot(r["normal_mean"], r["loop_mean"], mk, color="lightgray", ms=4)
        a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
        a.set_xlabel("Δ log p, true next token on normal segments"); a.set_title(f"{mode} (gray: random directions)")
    ax[0].set_ylabel("Δ log p, repeated token on loop segments"); ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{ROOT}/figs/e8_selectivity.png", dpi=120)


if __name__ == "__main__":
    main()
