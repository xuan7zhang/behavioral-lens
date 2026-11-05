"""E3: fit J-lens (exact gradient) and R-lens (lens.rlens_rules) on the BASE model, following vllm-lens PR #36
examples/jacobian_lens_fit.py (commit 20f7194): J_l = E[dh_target/dh_l], one forward of dim_batch copies per prompt and
d_model/dim_batch backwards with one-hot cotangents at the target over valid positions [skip_first, T-1), gradient at the
source averaged over the same positions. fp32 throughout.

Target = output of the LAST block = hs[24] (pre final norm), as in the reference (target_layer = n_layers - 1).
Sources = hs[1..23] (reference: every block output below the target). Readout of a lens: lm_head(final_norm(J_l h)).

python -m lens.e3_fit --rules gradient|lrp [--n_fineweb 1000 --n_seg 300] -> results/lenses/{jlens,rlens}_hs{k}.pt
"""
import argparse, json, os, random, time

import torch

import lens.model as LM
from lens import rlens_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET, SOURCES = 24, list(range(1, 24))


def fineweb(n):
    if n <= 1000 and os.path.exists(f"{ROOT}/data/fineweb_1000.jsonl"):
        return [json.loads(l) for l in open(f"{ROOT}/data/fineweb_1000.jsonl")][:n]     # first n of the cached 1000
    path = f"{ROOT}/data/fineweb_{n}.jsonl"
    if not os.path.exists(path):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        out = []
        for row in ds:
            if len(row["text"]) > 500:
                out.append(row["text"])
                if len(out) >= n:
                    break
        with open(path, "w") as f:
            for t in out:
                f.write(json.dumps(t) + "\n")
    return [json.loads(l) for l in open(path)]


def corpus(tok, n_fw, n_seg, max_len, seed=0):
    items = [("fineweb", tok(t, add_special_tokens=False).input_ids[:max_len]) for t in fineweb(n_fw)]
    S = [json.loads(l) for l in open(f"{ROOT}/results/data/segments.jsonl")]
    rng = random.Random(seed)
    for lab in ("loop", "normal"):
        pool = [s for s in S if s["split"] == "train" and s["label"] == lab and s["source"] in ("loop_base", "normal_pre_base", "normal_edited")]
        for s in rng.sample(pool, n_seg // 2):
            items.append((f"seg_{lab}", s["token_ids"][s["prompt_len"]:s["prompt_len"] + max_len]))
    return items


def fit(M, items, dim_batch, skip_first, rules):
    layers, d = M["layers"], M["cfg"].hidden_size
    for p in M["full"].parameters():
        p.requires_grad_(False)
    if rules in ("lrp", "lrp_ref"):
        print("R-lens rules installed on", rlens_rules.install(layers, mode="ref" if rules == "lrp_ref" else "full"), "modules", flush=True)
    acts = {}
    min_src = min(SOURCES)

    def mk(k):
        def hook(_m, _i, out):
            t = out if torch.is_tensor(out) else out[0]
            if k == min_src and not t.requires_grad:
                t.requires_grad_(True)
            acts[k] = t
        return hook
    hs = [layers[k - 1].register_forward_hook(mk(k)) for k in set(SOURCES) | {TARGET}]
    jac = {k: torch.zeros(2, d, d, dtype=torch.float64) for k in SOURCES}     # [even prompts, odd prompts] for split-half
    n_done, t0 = 0, time.time()
    try:
        for pi, (src, ids) in enumerate(items):
            T = len(ids)
            if T <= skip_first + 1:
                continue
            x = torch.tensor([ids], device="cuda").expand(dim_batch, -1)
            valid = torch.arange(skip_first, T - 1, device="cuda")
            b = torch.arange(dim_batch, device="cuda")
            acts.clear()
            with torch.enable_grad():
                M["lm"](input_ids=x, use_cache=False)
                target = acts[TARGET]
                srcs = [acts[k] for k in SOURCES]
                for s in srcs:
                    s.retain_grad()
                starts = list(range(0, d, dim_batch))
                for bi, st in enumerate(starts):
                    n = min(dim_batch, d - st)
                    for s in srcs:
                        s.grad = None
                    cot = torch.zeros_like(target)
                    cot[b[:n, None], valid[None, :], st + b[:n, None]] = 1.0
                    torch.autograd.backward(target, grad_tensors=cot, retain_graph=bi < len(starts) - 1, inputs=srcs)
                    for k, s in zip(SOURCES, srcs):
                        jac[k][n_done % 2, st:st + n] += s.grad[:n][:, valid, :].double().mean(1).cpu()
            n_done += 1
            if pi % 50 == 0:
                print(f"  {pi + 1}/{len(items)} ({src}, T={T}) {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in hs:
            h.remove()
        if rules in ("lrp", "lrp_ref"):
            rlens_rules.uninstall(layers)
    half = {k: float(torch.nn.functional.cosine_similarity(v[0].flatten(), v[1].flatten(), dim=0)) for k, v in jac.items()}
    return {k: (v.sum(0) / n_done).float() for k, v in jac.items()}, n_done, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", choices=("gradient", "lrp", "lrp_ref"), required=True)
    ap.add_argument("--n_fineweb", type=int, default=150)   # deviation: 1000 planned; 34 s/prompt with the torch DeltaNet path
    ap.add_argument("--n_seg", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--dim_batch", type=int, default=32)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = LM.tokenizer()
    M = LM.load(LM.BASE, dtype=torch.float32)
    items = corpus(tok, args.n_fineweb, args.n_seg, args.max_len)
    J, n, half = fit(M, items, args.dim_batch, args.skip_first, args.rules)
    name = args.tag or {"gradient": "jlens", "lrp": "rlens", "lrp_ref": "rlensref"}[args.rules]
    os.makedirs(f"{ROOT}/results/lenses", exist_ok=True)
    prov = dict(model=LM.BASE, rules=args.rules, n_prompts=n, n_fineweb=args.n_fineweb, n_seg=args.n_seg,
                ratio=f"{args.n_fineweb}:{args.n_seg} (fineweb:E2-train segments, half loop half normal)",
                max_len=args.max_len, skip_first=args.skip_first, dim_batch=args.dim_batch, target_hs=TARGET, source_hs=SOURCES,
                reference="vllm-lens PR #36 examples/jacobian_lens_fit.py @20f7194",
                split_half_cosine={int(k): v for k, v in half.items()})
    for k, m in J.items():
        torch.save(dict(J=m, hs=k, target_hs=TARGET, provenance=prov), f"{ROOT}/results/lenses/{name}_hs{k}.pt")
    json.dump(prov, open(f"{ROOT}/results/lenses/{name}_provenance.json", "w"), indent=1)
    print("saved", name, n, "prompts", flush=True)


if __name__ == "__main__":
    main()
