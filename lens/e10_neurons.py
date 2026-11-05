"""E10: the 64 edited neurons (Astra's) and the true top-64 by write score |u^T W0[:, j]|, blocks 15 and 19 (MLP writes hs[16] /
hs[20]). Neuron activation = act(gate) * up = input of down_proj. Base model, test-split segments.
Per neuron: mean activation on loop vs normal windows (per-segment mean), AUROC loop vs normal, signed write onto u
(u^T W0[:, j]), mean contribution to u (activation x write) on loop vs normal, top-activating tokens (20 strongest positions),
direct logit effect (W_U diag(1+w) W0[:, j]: top/bottom 10 tokens through the logit lens).
python -m lens.e10_neurons -> results/components/e10_neurons.json
"""
import json, os
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import lens.model as LM
from lens.lenses import segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = {15: 16, 19: 20}


def main():
    tok = LM.tokenizer()
    M = LM.load(LM.BASE)
    W = json.load(open(f"{ROOT}/results/components/weightdiff.json"))
    sets = {b: dict(astra=W[f"hs{h}"]["astra_neurons"], true_top64=W[f"hs{h}"]["true_top64"]) for b, h in BLOCKS.items()}
    neurons = {b: sorted(set(v["astra"]) | set(v["true_top64"])) for b, v in sets.items()}
    segs = [s for s in segments(split="test") if s["source"] in ("loop_base", "loop_base_late", "normal_pre_base", "normal_edited")]
    acts = {b: [] for b in BLOCKS}
    top = {b: {j: [] for j in neurons[b]} for b in BLOCKS}
    cap = {}
    hooks = [M["layers"][b].mlp.down_proj.register_forward_hook(lambda m, i, o, b=b: cap.__setitem__(b, i[0].detach())) for b in BLOCKS]
    with torch.no_grad():
        for si, s in enumerate(segs):
            ids = torch.tensor([s["token_ids"]], device="cuda")
            M["lm"](input_ids=ids, use_cache=False)
            a0 = s["prompt_len"]
            for b in BLOCKS:
                A = cap[b][0, a0:, neurons[b]].float()            # [T, n]
                acts[b].append(A.mean(0).cpu())
                mx, pos = A.max(0)
                for jj, j in enumerate(neurons[b]):
                    top[b][j].append((float(mx[jj]), si, int(pos[jj]) + a0))
            if si % 200 == 0:
                print(si, len(segs), flush=True)
    for h in hooks:
        h.remove()
    lab = np.array([s["label"] for s in segs])
    y = (lab == "loop").astype(int)
    WU = M["head"].weight.detach().float(); g = 1.0 + M["norm"].weight.detach().float()
    res = {"convention": "block 15 MLP -> hs[16], block 19 MLP -> hs[20]; activation = down_proj input", "n_segments": len(segs),
           "n_loop": int(y.sum()), "n_normal": int((1 - y).sum())}
    for b, h in BLOCKS.items():
        u = torch.from_numpy(np.load(f"{ROOT}/results/directions/weightdiff_hs{h}.npy")).float().cuda()
        W0 = M["layers"][b].mlp.down_proj.weight.detach().float()
        X = torch.stack(acts[b]).numpy()                          # [N, n]
        per = {}
        for jj, j in enumerate(neurons[b]):
            col = W0[:, j]
            w = float(u @ col)
            x = X[:, jj]
            auc = roc_auc_score(y, x)
            logit = WU @ (col * g)
            up_ids, dn_ids = torch.topk(logit, 10).indices.tolist(), torch.topk(-logit, 10).indices.tolist()
            ctx = []
            for v, si, p in sorted(top[b][j], reverse=True)[:20]:
                t = segs[si]["token_ids"]
                ctx.append(dict(act=v, label=segs[si]["label"], token=tok.decode([t[p]]), next=tok.decode([t[p + 1]]) if p + 1 < len(t) else "",
                                context=tok.decode(t[max(p - 8, 0):p + 1])))
            per[j] = dict(in_astra=j in sets[b]["astra"], in_true_top64=j in sets[b]["true_top64"], write_u=w,
                          mean_act_loop=float(x[y == 1].mean()), mean_act_normal=float(x[y == 0].mean()), auroc_loop=float(auc),
                          contrib_u_loop=float(w * x[y == 1].mean()), contrib_u_normal=float(w * x[y == 0].mean()),
                          logit_up=[tok.decode([i]) for i in up_ids], logit_down=[tok.decode([i]) for i in dn_ids],
                          top_contexts=ctx, top_ctx_loop_frac=float(np.mean([c["label"] == "loop" for c in ctx])))
        summ = {}
        for nm in ("astra", "true_top64"):
            js = sets[b][nm]
            P = [per[j] for j in js]
            dc = np.array([p["contrib_u_loop"] - p["contrib_u_normal"] for p in P])
            summ[nm] = dict(n=len(js), median_auroc=float(np.median([p["auroc_loop"] for p in P])),
                            frac_auroc_gt_0_7=float(np.mean([p["auroc_loop"] > 0.7 for p in P])),
                            frac_auroc_lt_0_3=float(np.mean([p["auroc_loop"] < 0.3 for p in P])),
                            sum_contrast_u=float(dc.sum()), sum_abs_write=float(np.sum([abs(p["write_u"]) for p in P])),
                            frac_more_active_on_loop=float(np.mean([p["mean_act_loop"] > p["mean_act_normal"] for p in P])),
                            top_ctx_loop_frac_mean=float(np.mean([p["top_ctx_loop_frac"] for p in P])),
                            common_top_tokens=Counter(c["token"] for p in P for c in p["top_contexts"]).most_common(15))
        only_a = sorted(set(sets[b]["astra"]) - set(sets[b]["true_top64"]))
        only_t = sorted(set(sets[b]["true_top64"]) - set(sets[b]["astra"]))
        for nm, js in (("astra_only", only_a), ("true_only", only_t)):
            P = [per[j] for j in js]
            summ[nm] = dict(n=len(js), median_auroc=float(np.median([p["auroc_loop"] for p in P])) if P else None,
                            sum_contrast_u=float(np.sum([p["contrib_u_loop"] - p["contrib_u_normal"] for p in P])),
                            sum_abs_write=float(np.sum([abs(p["write_u"]) for p in P])))
        res[f"block{b}"] = dict(summary=summ, neurons={str(j): v for j, v in per.items()})
        print(f"block {b} (hs[{h}]):", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "common_top_tokens"} for k, v in summ.items()}), flush=True)
        print("   astra common top tokens:", summ["astra"]["common_top_tokens"][:10], flush=True)
    json.dump(res, open(f"{ROOT}/results/components/e10_neurons.json", "w"), indent=1)


if __name__ == "__main__":
    main()
