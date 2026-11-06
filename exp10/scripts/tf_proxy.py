"""Exp10 addendum A: cheap proxies for every Qwen variant with a screening result (same 20 loop texts as tf_grid.py).
Proxies: Δ loop NLL, Δ other NLL, loop-position projection on u (hs16 / hs20), and the escape rate of short free-running
rollouts (4 × 256 tokens from each loop-onset prefix; T 0.6, top-p 0.95, top-k 20; HF has no presence penalty, so none)."""
import glob, gzip, json, os, sys
import numpy as np, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "scripts")
from astra_eval import build_prompt
R = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
VAR = {"base": f"{R}/base", "p10c": f"{R}/p10c", "p25c": f"{R}/p25c", "p50c": f"{R}/p50c"}
for d in sorted(glob.glob(f"{R}/variants/*")): VAR[os.path.basename(d)] = d
for d in sorted(glob.glob(f"{R}/exp10/*")): VAR["exp10_" + os.path.basename(d)] = d
tok = AutoTokenizer.from_pretrained(f"{R}/base")
model = AutoModelForCausalLM.from_pretrained(f"{R}/base", dtype=torch.bfloat16).to("cuda").eval()
params = dict(model.named_parameters())
def pname(k):
    for c in (k, k.replace("model.language_model.", "model.")):
        if c in params: return c
    raise KeyError(k)
base = load_file(f"{R}/base/model.safetensors"); DIRTY = set()
def set_variant(path):
    for k in list(DIRTY): params[pname(k)].data.copy_(base[k]); DIRTY.discard(k)
    if path.endswith("/base"): return 0
    sd = load_file(f"{path}/model.safetensors")
    for k in base:
        if k in sd and sd[k].shape == base[k].shape and not torch.equal(sd[k], base[k]): params[pname(k)].data.copy_(sd[k]); DIRTY.add(k)
    return len(DIRTY)
u = {L: torch.tensor(np.load(f"shared/u_L{L}.npy")).float().cuda() for L in (15, 19)}
texts = json.load(open("exp10/notes/tf_grid.json"))["texts"]
qs = {}
for l in open("data/gsm8k_test.jsonl"): r = json.loads(l); qs[r["id"]] = r["question"]
for l in open("data/math500.jsonl"): r = json.loads(l); qs[r["id"]] = r["problem"]
rows = {}
for f in glob.glob("astra_logs/exp8_math/eval/unedited/parts/*/*_first_samples.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        r = json.loads(l)
        if r["id"] in texts and r["id"] not in rows: rows[r["id"]] = r
def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m
T = []
for tid in texts:
    prompt = tok.apply_chat_template([{"role": "user", "content": build_prompt(qs[tid])}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(prompt).input_ids; ids = (pid + tok(rows[tid]["text"], add_special_tokens=False).input_ids)[:4096]; m = loop_mask(ids); m[:len(pid)] = False
    onset = int(np.nonzero(m)[0][0]); T.append((tid, ids, m, len(pid), onset))
@torch.no_grad()
def tf(ids, m, P):
    o = model.model(torch.tensor([ids]).cuda(), output_hidden_states=True); h = o.last_hidden_state[0]; tgt = torch.tensor(ids[1:]).cuda()
    pos = np.arange(len(ids) - 1); lp = pos[m[1:] & (pos >= P)]; op = pos[(~m[1:]) & (pos >= P)]
    def nll(p):
        out = []
        for s in range(0, len(p), 256):
            q = torch.tensor(p[s:s + 256]).cuda(); out.append(torch.nn.functional.cross_entropy(model.lm_head(h[q]).float(), tgt[q], reduction="none"))
        return torch.cat(out).mean().item()
    lpos = torch.tensor(np.nonzero(m)[0]).cuda()
    return nll(lp), nll(op), float((o.hidden_states[16][0].float()[lpos] @ u[15]).mean()), float((o.hidden_states[20][0].float()[lpos] @ u[19]).mean())
@torch.no_grad()
def escape(ids, onset, seed):
    torch.manual_seed(seed)
    pre = torch.tensor([ids[:onset]]).cuda()
    out = model.generate(pre, attention_mask=torch.ones_like(pre), do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=256,
                         num_return_sequences=4, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    esc, fr = [], []
    for s in out:
        full = s.tolist(); new = len(full) - onset; mm = loop_mask(full)[onset:]
        f = float(mm.mean()) if new > 0 else 0.0; fr.append(f); esc.append(f < 0.5)
    return float(np.mean(esc)), float(np.mean(fr))
outp = "exp10/notes/tf_proxy.json"; res = json.load(open(outp)) if os.path.exists(outp) else {}
for name, path in VAR.items():
    if name in res: continue
    n = set_variant(path); rec = dict(n_tensors=n)
    vals = [tf(ids, m, P) for _, ids, m, P, _ in T]
    rec.update(loop=float(np.mean([v[0] for v in vals])), other=float(np.mean([v[1] for v in vals])),
               proj_u15=float(np.mean([v[2] for v in vals])), proj_u19=float(np.mean([v[3] for v in vals])))
    es = [escape(ids, on, 17 * i) for i, (_, ids, _, _, on) in enumerate(T)]
    rec.update(escape=float(np.mean([e[0] for e in es])), rollout_loop_frac=float(np.mean([e[1] for e in es])))
    ed = [escape(ids, len(ids), 31 * i) for i, (_, ids, _, _, _) in enumerate(T)]   # deep in the loop: the 4,096-token end
    rec.update(escape_deep=float(np.mean([e[0] for e in ed])), rollout_loop_frac_deep=float(np.mean([e[1] for e in ed])))
    res[name] = rec; json.dump(res, open(outp, "w"), indent=1)
    print(f"{name:32s} n={n:2d} loop {rec['loop']:.4f} other {rec['other']:.4f} pu15 {rec['proj_u15']:+.2f} pu19 {rec['proj_u19']:+.2f} escape {rec['escape']:.2f} deep {rec['escape_deep']:.2f} rollout-loop {rec['rollout_loop_frac']:.2f} / {rec['rollout_loop_frac_deep']:.2f}", flush=True)
