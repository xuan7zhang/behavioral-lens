"""E7 / Q4: what does the edit suppress downstream? Base-fit lenses read base and edited on the SAME tokens (teacher forcing).
For hs[17..24] and each lens: change (edited - base) in log-probability of
  target  = the repeated token x[t+1] (loop positions, in the verbatim cycle) / the true next token (normal positions)
  think   = '</think>'
  others  = the top-100 base-probability tokens at that position, excluding target and '</think>' (mean)
Content-dependence check: for the same token ids that were targets, their change at positions of the same segment where they
are NOT the next token (target-as-non-target), and the 30 tokens suppressed most on average with how often each is the target.
Test split, up to 100 segments per label, 32 positions per segment (seed 0).
python -m lens.e7_edit_effect -> results/components/e7.json, figs/e7_edit_effect.png
"""
import json, os
from collections import Counter, defaultdict

import numpy as np
import torch

import lens.model as LM
from lens.lenses import LENSES, Lenses, loop_positions, normal_positions, sample, segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS, NSEG, NPOS = list(range(17, 25)), 100, 32


def main():
    tok = LM.tokenizer()
    think = tok.convert_tokens_to_ids("</think>")
    B, E = LM.load(LM.BASE), LM.load(LM.EDITED)
    LB = Lenses(B, hs=HS)
    rng = np.random.default_rng(0)
    res = {"convention": "hs index; edit sits in blocks 15/19 (writes hs[16], hs[20]); readouts at hs[17..24]"}
    for kind, sources in (("loop", ("loop_base", "loop_base_late")), ("normal", ("normal_pre_base", "normal_edited"))):
        segs = segments(split="test", sources=sources, labels=(kind,))
        segs = [segs[i] for i in sorted(rng.choice(len(segs), min(NSEG, len(segs)), replace=False))]
        acc = {(n, k): defaultdict(list) for n in LENSES for k in HS}
        mean_delta = {(n, k): None for n in LENSES for k in (20, 24)}
        tgt_counter = Counter()
        npos = 0
        with torch.no_grad():
            for si, s in enumerate(segs):
                pos = loop_positions(s) if kind == "loop" else normal_positions(s)
                pos = sample(pos, NPOS, seed=si)
                if not pos:
                    continue
                ids = torch.tensor([s["token_ids"]], device="cuda")
                _, hb = LM.forward(B, ids, keep=set(HS), logits=False)
                _, he = LM.forward(E, ids, keep=set(HS), logits=False)
                p = torch.tensor(pos, device="cuda")
                tgt = ids[0, p + 1]
                tgt_counter.update(tgt.tolist())
                # positions of the same segment whose next token is not the target id: for target-as-non-target
                other_pos = torch.tensor(sample(normal_positions(s), 64, seed=10_000 + si), device="cuda")
                for n in LENSES:
                    for k in HS:
                        zb, ze = LB.logits(n, k, hb[k][0, p]), LB.logits(n, k, he[k][0, p])   # base-fit lens on both models
                        lb, le = torch.log_softmax(zb, -1), torch.log_softmax(ze, -1)
                        d = le - lb
                        a = acc[(n, k)]
                        # softmax-invariant view: logit change minus the median logit change over the vocabulary
                        dz = ze - zb
                        dz = dz - dz.median(dim=-1, keepdim=True).values
                        a["centered_dlogit_target"] += dz.gather(1, tgt[:, None])[:, 0].tolist()
                        a["centered_dlogit_think"] += dz[:, think].tolist()
                        tz = torch.topk(zb, 102, dim=-1).indices
                        mz = (tz != tgt[:, None]) & (tz != think)
                        a["centered_dlogit_others_top100"] += ((dz.gather(1, tz) * mz).sum(1) / mz.sum(1)).tolist()
                        a["target"] += d.gather(1, tgt[:, None])[:, 0].tolist()
                        a["think"] += d[:, think].tolist()
                        top = torch.topk(lb, 102, dim=-1).indices
                        mask = (top != tgt[:, None]) & (top != think)
                        dd = d.gather(1, top)
                        a["others_top100"] += ((dd * mask).sum(1) / mask.sum(1)).tolist()
                        a["base_logprob_target"] += lb.gather(1, tgt[:, None])[:, 0].tolist()
                        # same token ids at positions where they are not the next token
                        lbo = torch.log_softmax(LB.logits(n, k, hb[k][0, other_pos]), -1)
                        leo = torch.log_softmax(LB.logits(n, k, he[k][0, other_pos]), -1)
                        nxt = ids[0, other_pos + 1]
                        u = torch.unique(tgt)
                        do = (leo - lbo)[:, u]
                        notnext = (nxt[:, None] != u[None, :])
                        if notnext.any():
                            a["target_ids_as_non_target"] += do[notnext].tolist()
                        if k in (20, 24):
                            v = d.double().sum(0)
                            mean_delta[(n, k)] = v if mean_delta[(n, k)] is None else mean_delta[(n, k)] + v
                npos += len(pos)
        out = {}
        for (n, k), a in acc.items():
            out[f"{n}/hs{k}"] = {key: dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v),
                                           se=float(np.std(v) / np.sqrt(max(len(v), 1)))) for key, v in a.items()}
        tops = {}
        for (n, k), v in mean_delta.items():
            v = (v / npos).float()
            idx = torch.topk(-v, 30).indices.tolist()
            tops[f"{n}/hs{k}"] = [dict(token=tok.decode([i]), mean_dlogprob=float(v[i]), frac_positions_where_target=tgt_counter[i] / npos) for i in idx]
        res[kind] = dict(n_positions=npos, n_segments=len(segs), by_lens_hs=out, most_suppressed=tops)
        for n in LENSES:
            r = out[f"{n}/hs24"]
            print(f"{kind:6s} {n:6s} hs24: Δlogp target {r['target']['mean']:+.3f} ±{r['target']['se']:.3f} | </think> {r['think']['mean']:+.3f} | "
                  f"others {r['others_top100']['mean']:+.3f} | target-ids-as-non-target {r['target_ids_as_non_target']['mean']:+.3f} || centered Δlogit: "
                  f"target {r['centered_dlogit_target']['mean']:+.3f} </think> {r['centered_dlogit_think']['mean']:+.3f} others {r['centered_dlogit_others_top100']['mean']:+.3f}", flush=True)
        print("   most suppressed (logit hs24):", [t["token"] for t in tops["logit/hs24"][:15]], flush=True)
    json.dump(res, open(f"{ROOT}/results/components/e7.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for a, n in zip(ax, LENSES):
        for kind, ls in (("loop", "-"), ("normal", "--")):
            for key, c in (("target", "C3"), ("think", "C2"), ("others_top100", "C7"), ("target_ids_as_non_target", "C0")):
                y = [res[kind]["by_lens_hs"][f"{n}/hs{k}"][key]["mean"] for k in HS]
                a.plot(HS, y, ls, color=c, label=f"{key} ({kind})")
        a.axhline(0, color="k", lw=0.5); a.set_title(f"{n}: Δ log p (edited − base)"); a.set_xlabel("hs[k]")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{ROOT}/figs/e7_edit_effect.png", dpi=120)


if __name__ == "__main__":
    main()
