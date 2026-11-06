"""Exp7 lens angle: fit J-lens and R-lens on the START Gemma 3 270M IT (PREREG_x7_circuit_lens.md, L5).
Recipe of vllm-lens PR #36 (examples/jacobian_lens_fit.py @20f7194): target = output of the last block (hs[18], pre final norm),
sources = outputs of blocks 0..16 (hs[1..17]); one forward of dim_batch copies per prompt, one-hot cotangents at the target on
positions [skip_first, T-1), source gradient averaged over the same positions; fp32. R-lens = the rules of the LessWrong post
(nv8oedrnLXKRzNEL9): LN-rule on the four residual-stream RMSNorms of each Gemma 3 block (input / post-attention / pre-feedforward /
post-feedforward; zero-centred weight, x * rsqrt * (1 + w)), identity rule on GELU-tanh, half rule on act(gate) * up; attention
and q/k norms untouched.
  python x7_lens_fit.py gradient|lrp  ->  exp7/lenses/{jlens,rlens}.pt  (+ provenance with split-half cosines)"""
import json, math, os, random, sys, time, types

import torch

import x7_local as L
from tk_lib import DEV, NL, load, ckpt, split, prompt_ids, answer_ids, TEMPLATE

rules = sys.argv[1]
TARGET, SOURCES, MAXLEN, SKIP, DIMB = NL, list(range(1, NL)), 128, 16, 16
OUT = os.path.join(L.lib.R, "lenses"); os.makedirs(OUT, exist_ok=True)


def _exact(t, s):
    return t.detach() + (s - s.detach())


def rms_lrp(self, x):
    xf = x.float()
    s = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    sur = (s * (1.0 + self.weight.float())).type_as(x)
    with torch.no_grad():
        true = self._r_orig(x)
    if not getattr(self, "_ok", False):
        assert float((sur - true).abs().max() / true.abs().max().clamp_min(1e-6)) < 1e-2, "RMSNorm surrogate mismatch"
        self._ok = True
    return _exact(true, sur)


def mlp_lrp(self, x):
    g, u = self.gate_proj(x), self.up_proj(x)
    c = math.sqrt(2.0 / math.pi)
    a = g * (0.5 * (1.0 + torch.tanh(c * (g + 0.044715 * g.pow(3))))).detach()      # identity rule, GELU-tanh
    sur = 0.5 * (a * u.detach() + a.detach() * u)                                     # half rule
    with torch.no_grad():
        true_h = self.act_fn(g) * u
    out = self.down_proj(_exact(true_h, sur))
    if not getattr(self, "_ok", False):
        with torch.no_grad():
            ref = self._r_orig(x)
        assert float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)) < 1e-2, "MLP surrogate mismatch"
        self._ok = True
    return out


def install(model):
    n = 0
    for lay in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            m = getattr(lay, name); m._r_orig = m.forward; m.forward = types.MethodType(rms_lrp, m); n += 1
        lay.mlp._r_orig = lay.mlp.forward; lay.mlp.forward = types.MethodType(mlp_lrp, lay.mlp); n += 1
    return n


def corpus(tok):
    fw = [json.loads(l) for l in open("/tealab-data/yush/lens_loop/data/fineweb_1000.jsonl")][:150]
    items = [("fineweb", tok(t, add_special_tokens=False)["input_ids"][:MAXLEN]) for t in fw]
    rng = random.Random(0)
    rows = split("100")["train"]
    for r in rng.sample(rows, 50):
        g = r["swap"] if rng.random() < 0.5 else r["correct"]
        ids = prompt_ids(tok, TEMPLATE.format(question=r["question"], guess=g)) + answer_ids(tok, r["correct"])
        items.append(("syc_train", ids[:MAXLEN]))
    return [x for x in items if len(x[1]) > SKIP + 2]


def main():
    torch.manual_seed(0)
    tok, M = load(ckpt("100", "base"))                  # the start checkpoint (shared by all pairs)
    for p in M.parameters():
        p.requires_grad_(False)
    if rules == "lrp":
        print("R-lens rules on", install(M), "modules", flush=True)
    layers, d = M.model.layers, M.config.hidden_size
    acts = {}

    def mk(k):
        def hook(_m, _i, out):
            t = out if torch.is_tensor(out) else out[0]
            if k == min(SOURCES) and not t.requires_grad:
                t.requires_grad_(True)
            acts[k] = t
        return hook
    hs = [layers[k - 1].register_forward_hook(mk(k)) for k in set(SOURCES) | {TARGET}]
    jac = {k: torch.zeros(2, d, d, dtype=torch.float64) for k in SOURCES}
    items = corpus(tok)
    n, t0 = 0, time.time()
    b = torch.arange(DIMB, device=DEV)
    for pi, (src, ids) in enumerate(items):
        T = len(ids)
        x = torch.tensor([ids], device=DEV).expand(DIMB, -1)
        valid = torch.arange(SKIP, T - 1, device=DEV)
        acts.clear()
        with torch.enable_grad():
            M(input_ids=x, use_cache=False)
            target, srcs = acts[TARGET], [acts[k] for k in SOURCES]
            for s in srcs:
                s.retain_grad()
            starts = list(range(0, d, DIMB))
            for bi, st in enumerate(starts):
                m = min(DIMB, d - st)
                for s in srcs:
                    s.grad = None
                cot = torch.zeros_like(target)
                cot[b[:m, None], valid[None, :], st + b[:m, None]] = 1.0
                torch.autograd.backward(target, grad_tensors=cot, retain_graph=bi < len(starts) - 1, inputs=srcs)
                for k, s in zip(SOURCES, srcs):
                    jac[k][n % 2, st:st + m] += s.grad[:m][:, valid, :].double().mean(1).cpu()
        n += 1
        if pi % 50 == 0:
            print(f"{pi + 1}/{len(items)} {src} T={T} {time.time() - t0:.0f}s", flush=True)
    for h in hs:
        h.remove()
    half = {k: float(torch.nn.functional.cosine_similarity(v[0].flatten(), v[1].flatten(), dim=0)) for k, v in jac.items()}
    J = {k: (v.sum(0) / n).float() for k, v in jac.items()}
    name = "jlens" if rules == "gradient" else "rlens"
    torch.save(dict(J=J, target_hs=TARGET, source_hs=SOURCES), f"{OUT}/{name}.pt")
    json.dump(dict(rules=rules, n_prompts=n, split_half_cosine=half, corpus="150 FineWeb (sample-10BT, cached) + 50 pair-100 train chat prompts",
                   max_len=MAXLEN, skip_first=SKIP, dim_batch=DIMB, reference="vllm-lens PR #36; R rules: LessWrong nv8oedrnLXKRzNEL9"),
              open(f"{OUT}/{name}_provenance.json", "w"), indent=1)
    print("saved", name, n, "min split-half", round(min(half.values()), 4), flush=True)


if __name__ == "__main__":
    main()
