"""Exp 9: train TopK SAEs (k=64) on Qwen3.5-0.8B base residual states after layer 15 (hidden state 16) or 19 (hidden state 20).
Unlabelled corpus: our screening generations (base thinking on / off, p50c; two random 2048-token windows per thinking-on sample,
one per thinking-off sample) + WikiText-103 train articles (2048-token windows) to ~17M tokens. Online: forward the base model
(stopped after the layer), buffer ~1.2M states on CPU, shuffle, step every SAE of this layer on the same batches.
  python train_saes.py <layer 15|19>  ->  /data2/ec2-user/exp9/sae_L<layer>_w<width>_s<seed>.pt"""
import json, math, os, random, sys, time, numpy as np, torch
sys.path.insert(0, "/data/behavioral-lens-fast/scripts")
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.backends.cuda.matmul.allow_tf32 = True
LAYER = int(sys.argv[1]); OUT = os.environ.get("SAE_OUT", "/data2/ec2-user/exp9"); DEV = "cuda"
SPECS = [(1024, 0), (4096, 0), (16384, 0), (16384, 1), (65536, 0)]
K, BATCH, WIN, SKIP = 64, 4096, 2048, 8
TOK_TARGET = int(os.environ.get("TOK_TARGET", 17_000_000))
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra/base"
tok = AutoTokenizer.from_pretrained(ROOT)
model = AutoModelForCausalLM.from_pretrained(ROOT, dtype=torch.bfloat16).to(DEV).eval()
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
probs = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/gsm8k_test.jsonl")}
probs.update({json.loads(l)["id"]: json.loads(l)["problem"] for l in open("data/math500.jsonl")})
rng = random.Random(1234)
docs = []  # list of token-id windows
for f, think, nwin in (("base", True, 2), ("p50c", True, 2), ("base_nothink", False, 1)):
    for l in open(f"notes/screen/{f}.jsonl"):
        r = json.loads(l)
        p = tok.apply_chat_template([{"role": "user", "content": probs[r["id"]] + "\n" + INSTR}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
        ids = tok(p + r["text"], add_special_tokens=False).input_ids
        if len(ids) <= WIN: docs.append(ids); continue
        starts = sorted(rng.sample(range(0, len(ids) - WIN), min(nwin, len(ids) - WIN)))
        docs += [ids[s:s + WIN] for s in starts]
ngen = sum(len(d) for d in docs)
wiki = [json.loads(l)["text"] for l in open("/data2/ec2-user/exp9/wikitext_train.jsonl")]; rng.shuffle(wiki)
wt = 0
for a in wiki:
    if ngen + wt >= TOK_TARGET * 1.03: break
    ids = tok(a, add_special_tokens=False).input_ids
    for s in range(0, len(ids) - 256, WIN):
        docs.append(ids[s:s + WIN]); wt += len(ids[s:s + WIN])
rng.shuffle(docs)
print(f"windows {len(docs)}; generation tokens {ngen/1e6:.1f}M, wikitext {wt/1e6:.1f}M", flush=True)


class Stop(Exception): pass
def states(batch_docs):
    L = max(len(d) for d in batch_docs); ids = torch.full((len(batch_docs), L), tok.pad_token_id or 0); att = torch.zeros_like(ids)
    for i, d in enumerate(batch_docs): ids[i, :len(d)] = torch.tensor(d); att[i, :len(d)] = 1
    cap = {}
    def hook(m, i, o):
        cap["x"] = (o[0] if isinstance(o, tuple) else o); raise Stop
    h = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        with torch.no_grad(): model.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV))
    except Stop: pass
    h.remove()
    x = cap["x"]; out = []
    for i, d in enumerate(batch_docs): out.append(x[i, SKIP:len(d)].to(torch.float16).cpu())
    return torch.cat(out)


class TopKSAE(torch.nn.Module):
    def __init__(self, d, m, seed):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        Wd = torch.randn(m, d, generator=g); Wd /= Wd.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(Wd); self.W_enc = torch.nn.Parameter(Wd.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(m)); self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.register_buffer("last_fired", torch.zeros(m))
    def encode_pre(self, x): return (x - self.b_dec) @ self.W_enc + self.b_enc
    def encode(self, x, k=K):
        pre = self.encode_pre(x); v, i = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
    def decode(self, f): return f @ self.W_dec + self.b_dec


def train():
    saes = {s: TopKSAE(1024, s[0], 100 + s[1]).to(DEV) for s in SPECS}
    opts = {s: torch.optim.Adam(saes[s].parameters(), lr=2e-4 * math.sqrt(16384 / s[0]), betas=(0.9, 0.999)) for s in SPECS}
    total_steps = TOK_TARGET // BATCH; step = 0; scale = None; seen = 0; t0 = time.time(); logs = {s: [] for s in SPECS}
    di = 0
    while di < len(docs) and step < total_steps:
        buf = []; nb = 0
        while di < len(docs) and nb < 1_200_000:
            b = docs[di:di + 8]; di += 8; x = states(b); buf.append(x); nb += len(x)
        X = torch.cat(buf); X = X[torch.randperm(len(X))]
        if scale is None:
            scale = float(math.sqrt(1024) / X[:200000].float().norm(dim=1).mean())
            mu = (X[:200000].float() * scale).mean(0).to(DEV)
            for s in SPECS: saes[s].b_dec.data.copy_(mu)
            Xev = X[:16384].float().to(DEV) * scale; X = X[16384:]
        for bstart in range(0, len(X) - BATCH + 1, BATCH):
            xb = X[bstart:bstart + BATCH].float().to(DEV) * scale
            frac = step / total_steps; lr_mult = min(1.0, (step + 1) / 100) * (1.0 if frac < 0.8 else max(0.0, (1 - frac) / 0.2))
            for s in SPECS:
                sae, opt = saes[s], opts[s]
                for gparam in opt.param_groups: gparam["lr"] = 2e-4 * math.sqrt(16384 / s[0]) * lr_mult
                pre = sae.encode_pre(xb); v, i = pre.topk(K, dim=-1); f = torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
                rec = f @ sae.W_dec + sae.b_dec; err = xb - rec
                loss = err.pow(2).sum(-1).mean() / (xb - xb.mean(0)).pow(2).sum(-1).mean()
                fired = (f > 0).any(0); sae.last_fired += BATCH; sae.last_fired[fired] = 0
                dead = sae.last_fired > 2_000_000
                if dead.any():  # AuxK: dead latents reconstruct the residual error
                    kd = min(512, int(dead.sum())); pre_d = pre.masked_fill(~dead, -1e9); vd, idd = pre_d.topk(kd, dim=-1)
                    fd = torch.zeros_like(pre).scatter_(-1, idd, torch.relu(vd)); ed = fd @ sae.W_dec
                    loss = loss + (1 / 32) * (err.detach() - ed).pow(2).sum(-1).mean() / err.detach().pow(2).sum(-1).mean().clamp_min(1e-6)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                with torch.no_grad(): sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True)
                del pre, v, i, f, rec, err, loss
            step += 1; seen += BATCH
            if step % 500 == 0 or step == total_steps:
                with torch.no_grad():
                    msg = []
                    for s in SPECS:
                        se = sum(float((Xev[c:c + 2048] - saes[s].decode(saes[s].encode(Xev[c:c + 2048]))).pow(2).sum()) for c in range(0, len(Xev), 2048))
                        fvu = se / float((Xev - Xev.mean(0)).pow(2).sum())
                        deadf = float((saes[s].last_fired > 2_000_000).float().mean()); logs[s].append({"step": step, "fvu": fvu, "dead": deadf})
                        msg.append(f"w{s[0]}s{s[1]} fvu {fvu:.3f} dead {deadf:.3f}")
                print(f"step {step}/{total_steps} tokens {seen/1e6:.1f}M {time.time()-t0:.0f}s | " + " | ".join(msg), flush=True)
            if step >= total_steps: break
    for s in SPECS:
        torch.save({"state": saes[s].state_dict(), "scale": scale, "k": K, "layer": LAYER, "hs": LAYER + 1, "width": s[0], "seed": s[1],
                    "log": logs[s], "tokens": seen}, f"{OUT}/sae_L{LAYER}_w{s[0]}_s{s[1]}.pt")
    print("saved", flush=True)


train()
