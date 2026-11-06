"""Exp10: teacher-forced grid over Astra's three Qwen runs and recombinations (P1-P6 of PREREG_exp10.md).
Metric as q4b: NLL of the actual next token at verbatim-loop positions (32-gram rule) and at the other generated positions,
20 looping unedited generations (10 GSM8K, 10 MATH-500), 4,096 tokens. Also: mean projection of loop / other states on u
(hidden state 16 on u_15, hidden state 20 on u_19)."""
import glob, gzip, json, sys
import numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "scripts")
from astra_eval import build_prompt
NT, MAXT = 10, 4096
R = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{R}/base")
model = AutoModelForCausalLM.from_pretrained(f"{R}/base", dtype=torch.bfloat16).to("cuda").eval()
params = dict(model.named_parameters())
def pname(k):
    for c in (k, k.replace("model.language_model.", "model."), k.replace("model.language_model.", "model.model.")):
        if c in params: return c
    raise KeyError(k)
base = load_file(f"{R}/base/model.safetensors")
runs = {}
for run in ("p10c", "p25c", "p50c"):
    ed = load_file(f"{R}/{run}/model.safetensors")
    runs[run] = {k: ed[k] for k in ed if k in base and ed[k].shape == base[k].shape and not torch.equal(ed[k], base[k])}
    print(run, len(runs[run]), "changed tensors", flush=True)
touched = sorted(set().union(*[set(v) for v in runs.values()]) | {f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)})
W0 = {k: base[k].float() for k in touched}
u = {L: torch.tensor(np.load(f"shared/u_L{L}.npy")).float() for L in (15, 19)}
acols = {L: torch.tensor(np.load(f"shared/astra_cols_L{L}.npy")) for L in (15, 19)}
DP = "model.language_model.layers.{}.mlp.down_proj.weight"
# p10c per-tensor directions (rank one, alpha 1.5)
v10 = {}
for k, W in runs["p10c"].items():
    U, S, Vh = torch.linalg.svd(W.float() - W0[k], full_matrices=False); v10[k] = U[:, 0]
def refl(W, v, a, cols=None):
    W = W.clone()
    if cols is None: return W - a * torch.outer(v, v @ W)
    W[:, cols] = W[:, cols] - a * torch.outer(v, v @ W[:, cols]); return W
def aligned(v, ref): return v if float(v @ ref) >= 0 else -v
d15 = aligned(v10[DP.format(15)], u[15]); d19 = aligned(v10[DP.format(19)], u[19])
def perp(v, ref): w = v - (v @ ref) * ref; return w / w.norm()
variants = {"base": {}}
variants["p50c"] = {k: v.float() for k, v in runs["p50c"].items()}
variants["p10c"] = {k: v.float() for k, v in runs["p10c"].items()}
variants["p25c"] = {k: v.float() for k, v in runs["p25c"].items()}
variants["p10c_L15L19_only"] = {k: v.float() for k, v in runs["p10c"].items() if ".layers.15." in k or ".layers.19." in k}
for a in (3, 6):
    variants[f"p10c_alpha{a}"] = {k: refl(W0[k], v10[k], a) for k in runs["p10c"]}
variants["p10cdir_p50crecipe"] = {DP.format(15): refl(W0[DP.format(15)], d15, 6, acols[15]), DP.format(19): refl(W0[DP.format(19)], d19, 6, acols[19])}
variants["p10cperp_p50crecipe"] = {DP.format(15): refl(W0[DP.format(15)], perp(d15, u[15]), 6, acols[15]), DP.format(19): refl(W0[DP.format(19)], perp(d19, u[19]), 6, acols[19])}
variants["u_allcols_alpha1.5"] = {DP.format(L): refl(W0[DP.format(L)], u[L], 1.5) for L in (15, 19)}
h7 = "model.language_model.layers.15.self_attn.o_proj.weight"
variants["p50c_plus_h7off"] = {**variants["p50c"], h7: variants["p25c"][h7]}
g = torch.Generator().manual_seed(0)
for r in range(3):
    variants[f"p10c_randdirs{r}"] = {k: refl(W0[k], (lambda x: x / x.norm())(torch.randn(1024, generator=g)), 1.5) for k in runs["p10c"]}
def set_w(ws):
    for k in touched:
        params[pname(k)].data.copy_((ws.get(k, W0[k])).to(torch.bfloat16))
# texts
qs = {}
for l in open("data/gsm8k_test.jsonl"):
    r = json.loads(l); qs[r["id"]] = r["question"]
for l in open("data/math500.jsonl"):
    r = json.loads(l); qs[r["id"]] = r["problem"]
def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        gg = tuple(ids[i - n:i]); m[i] = gg in seen; seen.add(gg)
    return m
texts = []
for setname in ("gsm8k", "math500"):
    rows = []
    for f in sorted(glob.glob(f"astra_logs/exp8_math/eval/unedited/parts/{setname}_s*/{setname}_first_samples.jsonl.gz")): rows += [json.loads(l) for l in gzip.open(f, "rt")]
    got = 0
    for r in rows:
        if "</think>" in r["text"] or r["id"] not in qs: continue
        prompt = tok.apply_chat_template([{"role": "user", "content": build_prompt(qs[r["id"]])}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pid = tok(prompt).input_ids; ids = (pid + tok(r["text"], add_special_tokens=False).input_ids)[:MAXT]; m = loop_mask(ids); m[:len(pid)] = False
        if m.mean() > 0.03: texts.append((r["id"], ids, m, len(pid))); got += 1
        if got >= NT: break
print("texts", len(texts), [t[0] for t in texts], flush=True)
@torch.no_grad()
def run_text(ids, lp, pp):
    o = model.model(torch.tensor([ids]).cuda(), output_hidden_states=True); hs = o.hidden_states; h = o.last_hidden_state[0]
    tgt = torch.tensor(ids[1:]).cuda()
    def nll(pos):
        out = []
        for s in range(0, len(pos), 512):
            p = torch.tensor(pos[s:s+512]).cuda(); out.append(torch.nn.functional.cross_entropy(model.lm_head(h[p]).float(), tgt[p], reduction="none"))
        return torch.cat(out).mean().item()
    proj = {}
    for L, hi in ((15, 16), (19, 20)):
        x = hs[hi][0].float(); pu = x @ u[L].cuda()
        proj[f"loop_u{L}"] = pu[torch.tensor(lp).cuda()].mean().item(); proj[f"other_u{L}"] = pu[torch.tensor(pp).cuda()].mean().item()
    return nll(lp), nll(pp), proj
res = {}
for name, ws in variants.items():
    set_w(ws); lo, pr, pj = [], [], []
    for tid, ids, m, P in texts:
        pos = np.arange(len(ids) - 1); lp = pos[m[1:] & (pos >= P)].tolist(); pp = pos[(~m[1:]) & (pos >= P)].tolist()
        a, b, c = run_text(ids, lp, pp); lo.append(a); pr.append(b); pj.append(c)
    res[name] = dict(loop=float(np.mean(lo)), other=float(np.mean(pr)), loop_per_text=lo, other_per_text=pr,
                     proj={k: float(np.mean([p[k] for p in pj])) for k in pj[0]})
    b = res["base"]
    print(f"{name:24s} loop {res[name]['loop']:.4f} ({res[name]['loop']-b['loop']:+.4f})  other {res[name]['other']:.4f} ({res[name]['other']-b['other']:+.4f})  "
          + " ".join(f"{k} {v:+.2f}" for k, v in res[name]["proj"].items()), flush=True)
    json.dump(dict(texts=[t[0] for t in texts], res=res), open("exp10/notes/tf_grid.json", "w"), indent=1)
b = res["base"]; e50 = res["p50c"]["loop"] - b["loop"]
print("\nshare of p50c loop effect:")
for k, v in res.items():
    dl, do = v["loop"] - b["loop"], v["other"] - b["other"]
    print(f"  {k:24s} {dl / e50:6.2f}   loop/other ratio {dl / do if abs(do) > 1e-6 else float('nan'):6.2f}")
