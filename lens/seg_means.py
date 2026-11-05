"""Per-segment mean residual (window tokens only, prompt excluded) at hs[0..24], base model, bf16 forward, fp32 accumulation.
Then fixes the sign of the E1 directions (configs/config.yaml e1.sign) and re-saves them.

python -m lens.seg_means -> results/data/seg_means_base.pt {ids, sources, labels, splits, means [N, 25, 1024] fp16}
                            results/directions/weightdiff_hs{16,20}.npy (re-signed), results/components/weightdiff_sign.json
"""
import json, os

import numpy as np
import torch

from lens.model import BASE, forward, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    S = [json.loads(l) for l in open(f"{ROOT}/results/data/segments.jsonl")]
    M = load(BASE)
    means = torch.zeros(len(S), 25, M["cfg"].hidden_size, dtype=torch.float16)
    with torch.no_grad():
        for i, s in enumerate(S):
            ids = torch.tensor([s["token_ids"]], device="cuda")
            _, hs = forward(M, ids, logits=False)
            a = s["prompt_len"] if s["source"] != "astra_pair" else 0
            for k in range(25):
                means[i, k] = hs[k][0, a:].float().mean(0).half().cpu()
            if i % 200 == 0:
                print(i, len(S), flush=True)
    torch.save(dict(ids=[s["id"] for s in S], sources=[s["source"] for s in S], labels=[s["label"] for s in S],
                    splits=[s["split"] for s in S], means=means), f"{ROOT}/results/data/seg_means_base.pt")
    lab = np.array([s["label"] for s in S]); spl = np.array([s["split"] for s in S]); src = np.array([s["source"] for s in S])
    out = {}
    for hs in (16, 20):
        u = torch.from_numpy(np.load(f"{ROOT}/results/directions/weightdiff_hs{hs}.npy")).float()
        m = means[:, hs].float()
        tr_loop, tr_norm = (lab == "loop") & (spl == "train"), (lab == "normal") & (spl == "train")
        d = m[torch.from_numpy(tr_loop)].mean(0) - m[torch.from_numpy(tr_norm)].mean(0)
        proj = float(u @ d)
        sign = 1.0 if proj > 0 else -1.0
        # Astra's own four matched pairs (fwd_4): repetitive minus productive, per case
        astra = [float(u @ (m[i + 1] - m[i])) for i in range(len(S)) if src[i] == "astra_pair" and S[i]["id"].startswith("astra_fwd4") and lab[i] == "normal"]
        np.save(f"{ROOT}/results/directions/weightdiff_hs{hs}.npy", (sign * u).numpy())
        out[f"hs{hs}"] = dict(proj_on_train_mean_diff_before=proj, sign_applied=sign, cos_u_meandiff=float(sign * proj / d.norm()),
                              astra_fwd4_rep_minus_prod_proj_after_sign=[sign * a for a in astra])
        print(f"hs{hs}: u.(loop-normal) = {proj:+.3f} -> sign {sign:+.0f}; cos(u, mean diff) = {out[f'hs{hs}']['cos_u_meandiff']:+.3f}; "
              f"Astra fwd4 pairs (after sign): {np.round(out[f'hs{hs}']['astra_fwd4_rep_minus_prod_proj_after_sign'], 3).tolist()}")
    json.dump(out, open(f"{ROOT}/results/components/weightdiff_sign.json", "w"), indent=1)


if __name__ == "__main__":
    main()
