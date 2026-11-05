"""E3 checks + the deferred E0 multi-hop sanity for J-/R-lens.
1. per-layer cosine between J-lens and R-lens: (a) matrix cosine, (b) mean per-token cosine of J_k h vs R_k h on 50 held-out
   FineWeb texts (prompts 900-949 of the cached 1000; never used in the fit), (c) top-1 agreement of their readouts
2. split-half cosine of each fit (from provenance)
3. multi-hop: rank of ' Japan' / ' Tokyo' per hs under each lens
python -m lens.e3_check -> results/lenses/e3_check.json, figs/e3_jr_cosine.png
"""
import json, os

import numpy as np
import torch

import lens.model as LM
from lens.e0_sanity import MULTIHOP
from lens.lenses import Lenses

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tok = LM.tokenizer()
    M = LM.load(LM.BASE, dtype=torch.float32)
    Ls = Lenses(M)
    res = {"convention": "hs index; target hs[24]; sources hs[1..23]"}
    res["matrix_cos"] = {k: float(torch.nn.functional.cosine_similarity(Ls.L[("jlens", k)].flatten(), Ls.L[("rlens", k)].flatten(), dim=0))
                         for k in range(1, 24)}
    for n in ("jlens", "rlens"):
        res[f"split_half_{n}"] = json.load(open(f"{ROOT}/results/lenses/{n}_provenance.json"))["split_half_cosine"]
    texts = [json.loads(l) for l in open(f"{ROOT}/data/fineweb_1000.jsonl")][900:950]
    cos = {k: [] for k in range(1, 24)}
    agree = {k: [] for k in range(1, 24)}
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([tok(t, add_special_tokens=False).input_ids[:128]], device="cuda")
            _, hs = LM.forward(M, ids, logits=False)
            for k in range(1, 24):
                h = hs[k][0, 16:]
                a, b = Ls.transport("jlens", k, h), Ls.transport("rlens", k, h)
                cos[k] += torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist()
                agree[k] += (Ls.logits("jlens", k, h).argmax(-1) == Ls.logits("rlens", k, h).argmax(-1)).float().tolist()
    res["token_cos_J_vs_R"] = {k: float(np.mean(v)) for k, v in cos.items()}
    res["top1_agree_J_vs_R"] = {k: float(np.mean(v)) for k, v in agree.items()}
    targets = {w: tok(w, add_special_tokens=False).input_ids[0] for w in (" Japan", " Tokyo")}
    res["multihop"] = {}
    with torch.no_grad():
        for p in MULTIHOP:
            ids = torch.tensor([tok(p).input_ids], device="cuda")
            _, hs = LM.forward(M, ids, logits=False)
            res["multihop"][p] = {}
            for n in ("logit", "jlens", "rlens"):
                for w, tid in targets.items():
                    r = []
                    for k in range(1, 25):
                        lg = Ls.logits(n, k, hs[k][0, -1])
                        r.append(int((lg > lg[tid]).sum()))
                    res["multihop"][p][f"{n}/{w.strip()}"] = r
    json.dump(res, open(f"{ROOT}/results/lenses/e3_check.json", "w"), indent=1)
    print("token cos J vs R by hs:", {k: round(v, 2) for k, v in res["token_cos_J_vs_R"].items()})
    print("split-half J:", {k: round(v, 3) for k, v in res["split_half_jlens"].items() if int(k) % 4 == 0})
    print("split-half R:", {k: round(v, 3) for k, v in res["split_half_rlens"].items() if int(k) % 4 == 0})
    for p, d in res["multihop"].items():
        print(p)
        for key, r in d.items():
            first = next((k + 1 for k, x in enumerate(r) if x < 10), None)
            print(f"   {key:14s} first hs with rank<10: {first}  ranks {r}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = list(range(1, 24))
    plt.figure(figsize=(7, 4))
    plt.plot(ks, [res["token_cos_J_vs_R"][k] for k in ks], "o-", label="per-token cos(J h, R h)")
    plt.plot(ks, [res["matrix_cos"][k] for k in ks], "s--", label="matrix cos(J, R)")
    plt.plot(ks, [res["top1_agree_J_vs_R"][k] for k in ks], "^:", label="top-1 readout agreement")
    for k in (4, 8, 12, 16, 20):
        plt.axvline(k, color="gray", lw=0.5)
    plt.xlabel("source hs[k] (gray: hs after a full-attention block)"); plt.legend(fontsize=8); plt.title("J-lens vs R-lens (base, held-out FineWeb)")
    plt.tight_layout(); plt.savefig(f"{ROOT}/figs/e3_jr_cosine.png", dpi=120)


if __name__ == "__main__":
    main()
