"""E0: environment and sanity checks. python -m lens.e0_sanity -> results/e0_sanity.json

1. both models load; only the two edited tensors differ
2. linear attention (Gated DeltaNet) forward + backward run (torch reference path; no fla installed) and the backward
   matches a finite-difference directional derivative
3. logit lens on hs[24] (model's own final norm + lm_head) reproduces the model logits
4. multi-hop prompt: logit-lens rank of ' Japan' / ' Tokyo' per layer (J-/R-lens versions run after E3)
"""
import json, os, platform, subprocess

import lens.model  # noqa: F401  (sets HF_HOME before any huggingface import)

import numpy as np
import torch
import transformers

from lens.model import BASE, EDITED, PREFIX, SUFFIX, forward, load, readout, tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTIHOP = ["The capital of the country where sushi originates is", "Sushi originates from the country whose capital city is"]


def main():
    torch.manual_seed(0)
    res = dict(versions=dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                             git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None))
    tok = tokenizer()
    B = load(BASE)
    E = load(EDITED)
    sb, se = B["full"].state_dict(), E["full"].state_dict()
    diff = [k for k in sb if not torch.equal(sb[k], se[k])]
    res["differing_tensors"] = diff
    res["layer_types"] = list(B["cfg"].layer_types)
    print("differing tensors:", diff)
    del E, se
    torch.cuda.empty_cache()
    # 3. logit lens on hs[24] == logits (bf16 forward, and fp32 forward)
    ids = tok(PREFIX + "What is 12 * 13?" + SUFFIX + "12 * 13 = 156. Wait, let me check: 12 * 13 = 156.", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        for dt in (torch.bfloat16, torch.float32):
            M = B if dt == torch.bfloat16 else load(BASE, torch.float32)
            lg, hs = forward(M, ids)
            ll = readout(M, hs[24])
            res[f"logitlens_hs24_maxabs_{str(dt)[6:]}"] = float((ll.float() - lg.float()).abs().max())
            res[f"n_hs_{str(dt)[6:]}"] = len(hs)
    print("hs24 logit lens vs logits max|diff|:", res["logitlens_hs24_maxabs_bfloat16"], res["logitlens_hs24_maxabs_float32"])
    # 2. backward through DeltaNet (fp32): gradient of a scalar at hs[23] w.r.t. hs[1], checked by finite difference
    F = load(BASE, torch.float32)
    ids2 = ids[:, :64]
    with torch.no_grad():
        _, hs = forward(F, ids2, keep={1})
    h1 = hs[1].detach().clone()
    v = torch.randn_like(h1); v = v / v.norm()
    w = torch.randn(h1.shape[-1], device="cuda")

    def f(x):
        _, hh = forward(F, ids2, keep={23}, edits={1: lambda h: x}, logits=False)
        return (hh[23][0, -1] * w).sum()
    x = h1.clone().requires_grad_(True)
    y = f(x)
    y.backward()
    g = float((x.grad * v).sum())
    eps = 1e-2
    with torch.no_grad():
        fd = float((f(h1 + eps * v) - f(h1 - eps * v)) / (2 * eps))
    res["deltanet_backward"] = dict(grad_dot_v=g, finite_diff=fd, rel_err=abs(g - fd) / max(abs(fd), 1e-8), grad_finite=bool(torch.isfinite(x.grad).all()))
    print("backward check:", res["deltanet_backward"])
    # 4. multi-hop: logit-lens rank of answer tokens per layer
    targets = {w: tok(w, add_special_tokens=False).input_ids for w in (" Japan", " Tokyo")}
    res["multihop_token_ids"] = targets
    res["multihop"] = {}
    with torch.no_grad():
        for p in MULTIHOP:
            ids3 = tok(p, return_tensors="pt").input_ids.cuda()
            _, hs = forward(F, ids3)
            rows = {}
            for name, tid in targets.items():
                r = []
                for k in range(25):
                    lg = readout(F, hs[k][0, -1])
                    r.append(int((lg > lg[tid[0]]).sum()))
                rows[name] = r
            top = tok.decode(readout(F, hs[24][0, -1]).argmax())
            res["multihop"][p] = dict(rank_by_hs=rows, top1_final=top)
            print(p, "| final top1:", repr(top))
            for name, r in rows.items():
                print(f"   {name:7s} rank by hs[0..24]: {r}")
    json.dump(res, open(f"{ROOT}/results/e0_sanity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
