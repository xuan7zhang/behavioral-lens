"""Exp 9 analysis (after train_saes.py for layers 15 and 19). Pre-registration: exp9/notes/PREREG_exp9_sae.md.
Eval texts: Astra's unedited GSM8K / MATH-500 first samples (not in the SAE training data), first 4096 tokens; 100 loop texts
(>3 % positions in 32-gram verbatim repeats, never closed) and 100 normal texts (closed thinking, <0.5 % loop positions).
Loop positions = repeat mask in loop texts; normal positions = all generated positions of normal texts.
  python analyze.py  ->  exp9/notes/exp9_results.json"""
import glob, gzip, json, math, random, re, sys, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import average_precision_score
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts"); import llm
sys.path.insert(0, "/data/behavioral-lens-fast/exp9/scripts")
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"; OUT = "/data2/ec2-user/exp9"; N = "/data/behavioral-lens-fast/exp9/notes"
ROOT = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
tok = AutoTokenizer.from_pretrained(f"{ROOT}/base")
model = AutoModelForCausalLM.from_pretrained(f"{ROOT}/base", dtype=torch.bfloat16).to(DEV).eval()
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
probs = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/gsm8k_test.jsonl")}
probs.update({json.loads(l)["id"]: json.loads(l)["problem"] for l in open("data/math500.jsonl")})
U = {15: torch.tensor(np.load("shared/u_L15.npy")).float(), 19: torch.tensor(np.load("shared/u_L19.npy")).float()}
U = {L: (v / v.norm()).to(DEV) for L, v in U.items()}
WID = [(1024, 0), (4096, 0), (16384, 0), (16384, 1), (65536, 0)]
rng = random.Random(0)


def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m


rows = []
for f in sorted(glob.glob("astra_logs/exp8_math/eval/unedited/parts/gsm8k_s*/gsm8k_first_samples.jsonl.gz")) + sorted(glob.glob("astra_logs/exp8_math/eval/unedited/parts/math500_s*/math500_first_samples.jsonl.gz")):
    rows += [json.loads(l) for l in gzip.open(f, "rt")]
rng.shuffle(rows)
loop_t, norm_t = [], []
for r in rows:
    if r["id"] not in probs: continue
    p = tok.apply_chat_template([{"role": "user", "content": probs[r["id"]] + "\n" + INSTR}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(p, add_special_tokens=False).input_ids; ids = (pid + tok(r["text"], add_special_tokens=False).input_ids)[:4096]
    m = loop_mask(ids); m[:len(pid)] = False; frac = m[len(pid):].mean()
    if "</think>" not in r["text"] and frac > 0.03 and len(loop_t) < 100: loop_t.append((ids, m, len(pid)))
    elif "</think>" in r["text"] and frac < 0.005 and len(norm_t) < 100: norm_t.append((ids, np.zeros(len(ids), bool), len(pid)))
    if len(loop_t) >= 100 and len(norm_t) >= 100: break
wiki = [json.loads(l)["text"] for l in open(f"{OUT}/wikitext_validation.jsonl")]
wiki_t = [(tok(a, add_special_tokens=False).input_ids[:2048], None, 0) for a in wiki]
print("eval texts: loop", len(loop_t), "normal", len(norm_t), "wiki", len(wiki_t), flush=True)


class Stop(Exception): pass
@torch.no_grad()
def states(ids, layers=(15, 19)):
    cap = {}
    def mk(L):
        def h(m, i, o):
            cap[L] = (o[0] if isinstance(o, tuple) else o)[0].float()
            if L == max(layers): raise Stop
        return h
    hs = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]
    try: model.model(input_ids=torch.tensor([ids]).to(DEV))
    except Stop: pass
    for h in hs: h.remove()
    return {L: cap[L].half().cpu() for L in layers}
S = {"loop": [states(t[0]) for t in loop_t], "norm": [states(t[0]) for t in norm_t], "wiki": [states(t[0]) for t in wiki_t]}
def pos_sel(kind, i):
    ids, m, P = (loop_t if kind == "loop" else norm_t)[i]
    sel = np.arange(len(ids)); return sel[m] if kind == "loop" else sel[P:]
print("states done", flush=True)


def load_sae(L, w, s):
    ck = torch.load(f"{OUT}/sae_L{L}_w{w}_s{s}.pt", map_location=DEV)
    st = ck["state"]; return {k: v.float() for k, v in st.items()}, ck["scale"], ck["k"], ck["log"][-1]
def enc(sd, x, k):
    pre = (x - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"]; v, i = pre.topk(k, dim=-1)
    return torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))


def omp(D, u, kmax=50):
    sel, res, cs = [], u.clone(), []
    for _ in range(kmax):
        j = int((D @ res).abs().argmax()); sel.append(j)
        A = D[sel].T; coef = torch.linalg.lstsq(A, u[:, None]).solution[:, 0]; rec = A @ coef
        res = u - rec; cs.append(float(rec @ u / rec.norm()))
    return sel, cs


R = {"n_loop_texts": len(loop_t), "n_norm_texts": len(norm_t)}
interp_sets = {}
for L in (15, 19):
    u = U[L]; R[L] = {}
    for (w, s) in WID:
        sd, scale, k, last = load_sae(L, w, s); D = sd["W_dec"]  # [m, d] unit rows
        cos = D @ u
        sel, cs = omp(D, u)
        sums = {c: torch.zeros(w, device=DEV) for c in ("loop", "norm")}; cnt = {c: 0 for c in sums}
        ru = {c: 0.0 for c in sums}; freq = {c: torch.zeros(w, device=DEV) for c in sums}
        fvu_n = {c: [0.0, 0.0] for c in sums}
        for c in ("loop", "norm"):
            for i, st in enumerate(S[c]):
                p = pos_sel(c, i)
                if len(p) == 0: continue
                x = st[L][p].float().to(DEV); xs = x * scale
                for ch in range(0, len(xs), 2048):
                    f = enc(sd, xs[ch:ch + 2048], k); sums[c] += f.sum(0); freq[c] += (f > 0).float().sum(0)
                    rec = f @ D + sd["b_dec"]; fvu_n[c][0] += float((xs[ch:ch + 2048] - rec).pow(2).sum())
                ru[c] += float((x @ u).sum()); cnt[c] += len(p); fvu_n[c][1] += float((xs - xs.mean(0)).pow(2).sum())
        mean = {c: sums[c] / cnt[c] for c in sums}
        contrib = (mean["loop"] - mean["norm"]) * cos / scale  # in raw units of r.u
        d_ru = ru["loop"] / cnt["loop"] - ru["norm"] / cnt["norm"]
        pos = contrib.clamp_min(0); tot = float(pos.sum()); order = torch.argsort(-contrib)
        cum = torch.cumsum(contrib[order].clamp_min(0), 0) / tot
        need = lambda q: int((cum < q).sum()) + 1
        r = {"fvu_train_last": last, "fvu_loop": fvu_n["loop"][0] / fvu_n["loop"][1], "fvu_norm": fvu_n["norm"][0] / fvu_n["norm"][1],
             "max_abs_cos": float(cos.abs().max()), "argmax_cos": int(cos.abs().argmax()), "n_cos_gt_0.3": int((cos.abs() > 0.3).sum()), "n_cos_gt_0.5": int((cos.abs() > 0.5).sum()),
             "omp_cos": cs, "omp_atoms": sel[:10], "omp_k_for_0.8": next((i + 1 for i, x in enumerate(cs) if x >= 0.8), None), "omp_k_for_0.9": next((i + 1 for i, x in enumerate(cs) if x >= 0.9), None),
             "delta_ru_loop_minus_norm": d_ru, "sum_contrib": float(contrib.sum()), "sum_pos_contrib": tot,
             "top1_share": float(contrib[order[0]] / tot), "top5_share": float(contrib[order[:5]].clamp_min(0).sum() / tot), "n_for_50": need(0.5), "n_for_80": need(0.8),
             "top_loop_side": [{"latent": int(j), "contrib": float(contrib[j]), "share": float(contrib[j] / tot), "cos_u": float(cos[j]), "freq_loop": float(freq["loop"][j] / cnt["loop"]), "freq_norm": float(freq["norm"][j] / cnt["norm"])} for j in order[:10].tolist()],
             "top_norm_side": [{"latent": int(j), "contrib": float(contrib[j]), "cos_u": float(cos[j]), "freq_loop": float(freq["loop"][j] / cnt["loop"]), "freq_norm": float(freq["norm"][j] / cnt["norm"])} for j in torch.argsort(contrib)[:5].tolist()],
             "alive_frac": float(((freq["loop"] + freq["norm"]) > 0).float().mean())}
        R[L][f"w{w}_s{s}"] = r
        print(f"L{L} w{w} s{s}: fvu loop {r['fvu_loop']:.3f} norm {r['fvu_norm']:.3f} | max|cos| {r['max_abs_cos']:.3f} (n>0.5: {r['n_cos_gt_0.5']}) OMP k(0.9) {r['omp_k_for_0.9']} | "
              f"delta r.u {d_ru:.2f} = sum contrib {r['sum_contrib']:.2f} + err | top1 {r['top1_share']:.2f} top5 {r['top5_share']:.2f} n50 {r['n_for_50']} n80 {r['n_for_80']} | top1 cos {r['top_loop_side'][0]['cos_u']:.2f}", flush=True)
        n_top = 8 if (w, s) == (16384, 0) else 4
        interp_sets[(L, w, s)] = [("loop_side", e["latent"]) for e in r["top_loop_side"][:n_top]] + ([("norm_side", e["latent"]) for e in r["top_norm_side"][:4]] if (w, s) == (16384, 0) else [])
        if (w, s) == (16384, 0):
            alive = torch.nonzero((freq["loop"] + freq["norm"]) > 0)[:, 0].tolist(); chosen = {j for _, j in interp_sets[(L, w, s)]}
            interp_sets[(L, w, s)] += [("random", j) for j in random.Random(L).sample([a for a in alive if a not in chosen], 10)]
    # cross-width: top loop-side latent of each SAE vs the 16k s0 top-8
    D16 = load_sae(L, 16384, 0)[0]["W_dec"]; top16 = [e["latent"] for e in R[L]["w16384_s0"]["top_loop_side"][:8]]
    R[L]["cross_width"] = {}
    for (w, s) in WID:
        Dw = load_sae(L, w, s)[0]["W_dec"]; j = R[L][f"w{w}_s{s}"]["top_loop_side"][0]["latent"]
        R[L]["cross_width"][f"w{w}_s{s}"] = {"top1": j, "cos_with_16k_top8": [round(float(Dw[j] @ D16[t]), 3) for t in top16]}
    print(f"L{L} cross-width (top-1 latent of each SAE vs 16k s0 top-8):", {k: v["cos_with_16k_top8"][:4] for k, v in R[L]["cross_width"].items()}, flush=True)

# ---------------- blind auto-interp
CATS = ["verbatim repetition/copying", "restating or re-reading the problem", "doubt/self-check", "wrap-up/final answer", "maths content", "formatting/whitespace", "other"]
docs_all = [("loop", i) for i in range(len(loop_t))] + [("norm", i) for i in range(len(norm_t))] + [("wiki", i) for i in range(len(wiki_t))]
def doc_ids(kind, i): return (loop_t if kind == "loop" else norm_t if kind == "norm" else wiki_t)[i][0]
def window(ids, t, left=32, right=4):
    a, b = max(0, t - left), min(len(ids), t + right + 1)
    w = [tok.decode([x]) for x in ids[a:b]]; w[t - a] = "[[" + w[t - a] + "]]"; return "".join(w).replace("\n", "\\n")
results_ai = []
for (L, w, s), latents in interp_sets.items():
    sd, scale, k, _ = load_sae(L, w, s); js = torch.tensor([j for _, j in latents], device=DEV)
    acts = []  # per doc: [T, n_latents]
    for kind, i in docs_all:
        x = S[kind][i][L].float().to(DEV) * scale
        a = torch.cat([enc(sd, x[ch:ch + 2048], k)[:, js] for ch in range(0, len(x), 2048)])
        acts.append(a.cpu())
    jobs = []
    for li, (side, j) in enumerate(latents):
        top = []
        for di, (kind, i) in enumerate(docs_all):
            a = acts[di][:, li]
            if a.max() <= 0: continue
            vals, idx = a.topk(min(2, len(a)))
            top += [(float(v), di, int(t)) for v, t in zip(vals, idx) if v > 0]
        top.sort(reverse=True); top = top[:40]
        if len(top) < 20: continue
        rr = random.Random(j); pick = top[:]; rr.shuffle(pick)
        expl = [window(doc_ids(*docs_all[di]), t) for _, di, t in pick[:12]]
        pos = [window(doc_ids(*docs_all[di]), t) for _, di, t in pick[12:20]]
        neg = []
        while len(neg) < 8:
            di = rr.randrange(len(docs_all)); a = acts[di][:, li]; t = rr.randrange(len(a))
            if a[t] == 0: neg.append(window(doc_ids(*docs_all[di]), t))
        jobs.append({"L": L, "w": w, "s": s, "side": side, "latent": j, "expl": expl, "score": [(x, 1) for x in pos] + [(x, 0) for x in neg]})
    def explain(job):
        return llm.chat("Below are snippets of text. In each, one internal feature of a language model is active on the token marked [[like this]] "
                        "(the snippets come from maths solutions written by the model, and from Wikipedia).\n\n" + "\n".join(f"- {e}" for e in job["expl"]) +
                        "\n\nIn one sentence, describe what this feature responds to. Reply with the sentence only.", max_tokens=90).strip()
    with ThreadPoolExecutor(8) as ex: descs = list(ex.map(explain, jobs))
    for job, d in zip(jobs, descs): job["description"] = d
    def score(args):
        job, d = args; items = job["score"][:]; random.Random(job["latent"] + 7).shuffle(items)
        out = llm.chat(f"A feature of a language model is described as: \"{d}\"\nFor each snippet below, rate 0-10 how strongly the feature is active on the "
                       f"marked [[token]]. Reply with {len(items)} integers separated by commas, nothing else.\n\n" + "\n".join(f"{n + 1}. {x}" for n, (x, _) in enumerate(items)), max_tokens=80)
        nums = [int(x) for x in re.findall(r"\d+", out)][:len(items)]; nums += [0] * (len(items) - len(nums))
        return float(average_precision_score([y for _, y in items], nums))
    pairs = [(job, job["description"]) for job in jobs] + [(job, jobs[(n + 5) % len(jobs)]["description"]) for n, job in enumerate(jobs)]
    with ThreadPoolExecutor(8) as ex: aps = list(ex.map(score, pairs))
    for n, job in enumerate(jobs): job["AP"], job["AP_shuffled"] = aps[n], aps[len(jobs) + n]
    results_ai += jobs
    print(f"auto-interp L{L} w{w} s{s}: {len(jobs)} latents", flush=True)
labs = [j["description"] for j in results_ai]
def classify(chunk):
    out = llm.chat("Assign each feature description below to exactly one category: " + "; ".join(f"({c + 1}) {x}" for c, x in enumerate(CATS)) +
                   ". Reply with one line per description: '<number>: <category number>'.\n\n" + "\n".join(f"{n + 1}. {d}" for n, d in enumerate(chunk)), max_tokens=12 * len(chunk) + 20)
    m = dict((int(a), int(b)) for a, b in re.findall(r"(\d+)\s*:\s*\(?(\d)", out)); return [CATS[m.get(n + 1, 7) - 1] for n in range(len(chunk))]
cats = []
for c in range(0, len(labs), 25): cats += classify(labs[c:c + 25])
for j, c in zip(results_ai, cats): j["category"] = c; j.pop("expl"); j.pop("score")
R["autointerp"] = results_ai
for L in (15, 19):
    for side in ("loop_side", "norm_side", "random"):
        x = [j for j in results_ai if j["L"] == L and j["w"] == 16384 and j["s"] == 0 and j["side"] == side]
        if x: print(f"L{L} 16k {side}: AP {np.mean([j['AP'] for j in x]):.2f} shuffled {np.mean([j['AP_shuffled'] for j in x]):.2f}; categories {sorted({j['category'] for j in x})}", flush=True)
json.dump(R, open(f"{N}/exp9_results.json", "w"), indent=1, default=str)

# ---------------- causal: Astra's rule with SAE directions (teacher-forced loop / normal loss, q4b proxy)
from safetensors.torch import load_file
base = load_file(f"{ROOT}/base/model.safetensors"); NAME = "model.language_model.layers.{}.mlp.down_proj.weight"
W0 = {L: base[NAME.format(L)].float() for L in (15, 19)}; cols = {L: np.load(f"shared/astra_cols_L{L}.npy") for L in (15, 19)}
def edited(L, d, alpha=6):
    d = (d / d.norm()).cpu(); W = W0[L].clone(); c = torch.tensor(cols[L]); W[:, c] = W0[L][:, c] - alpha * torch.outer(d, d @ W0[L][:, c]); return W
def set_w(ws):
    for L, W in ws.items(): model.model.layers[L].mlp.down_proj.weight.data.copy_(W.to(torch.bfloat16))
tf = [t for t in loop_t[:6]]
@torch.no_grad()
def nll_at(ids, positions):
    h = model.model(torch.tensor([ids]).to(DEV)).last_hidden_state[0]; tgt = torch.tensor(ids[1:]).to(DEV); out = []
    for s0 in range(0, len(positions), 512):
        p = torch.tensor(positions[s0:s0 + 512]).to(DEV); out.append(torch.nn.functional.cross_entropy(model.lm_head(h[p]).float(), tgt[p], reduction="none"))
    return torch.cat(out)
def evaluate(ws):
    set_w(ws); lo, pr = [], []
    for ids, m, P in tf:
        pos = np.arange(len(ids) - 1); lp = pos[m[1:] & (pos >= P)]; pp = pos[(~m[1:]) & (pos >= P)]
        lo.append(nll_at(ids, lp.tolist()).mean().item()); pr.append(nll_at(ids, pp.tolist()).mean().item())
    return float(np.mean(lo)), float(np.mean(pr))
dirs = {}
for (w, s) in WID:
    D = {L: load_sae(L, w, s)[0]["W_dec"] for L in (15, 19)}
    t1 = {L: R[L][f"w{w}_s{s}"]["top_loop_side"][0]["latent"] for L in (15, 19)}
    dirs[f"top1_w{w}_s{s}"] = {L: D[L][t1[L]] for L in (15, 19)}
    if (w, s) == (16384, 0):
        for kk in (5, 10):
            tops = {L: R[L]["w16384_s0"]["top_loop_side"][:kk] for L in (15, 19)}
            dirs[f"top{kk}_weighted_w16k"] = {L: sum(e["contrib"] * torch.sign(torch.tensor(e["cos_u"])) * D[L][e["latent"]] for e in tops[L]) for L in (15, 19)}
            # contribution * sign(cos) aligns each decoder with +u before summing
        dirs["omp10_w16k"] = {}
        for L in (15, 19):
            sel = R[L]["w16384_s0"]["omp_atoms"]; A = D[L][sel].T; coef = torch.linalg.lstsq(A, U[L][:, None]).solution[:, 0]; dirs["omp10_w16k"][L] = A @ coef
        g = torch.Generator().manual_seed(0)
        for i in range(5): dirs[f"random_latent{i}"] = {L: D[L][int(torch.randint(0, 16384, (1,), generator=g))] for L in (15, 19)}
dirs["u"] = {L: U[L] for L in (15, 19)}
res_e = {}
lo0, pr0 = evaluate({L: W0[L] for L in W0}); res_e["base"] = {"loop": lo0, "prod": pr0}
for name, d in dirs.items():
    lo, pr = evaluate({L: edited(L, d[L]) for L in (15, 19)})
    res_e[name] = {"loop": lo, "prod": pr, "d_loop": lo - lo0, "d_prod": pr - pr0, "cos_u": {L: float(d[L] @ U[L] / d[L].norm()) for L in (15, 19)}}
for name in dirs: res_e[name]["frac_of_u_loop_effect"] = res_e[name]["d_loop"] / res_e["u"]["d_loop"]
for name, v in res_e.items():
    if name != "base": print(f"edit {name:24s} d_loop {v['d_loop']:+.4f} d_prod {v['d_prod']:+.4f} frac_of_u {v['frac_of_u_loop_effect']:+.2f} cos_u {v['cos_u']}", flush=True)
R["causal_tf"] = res_e
json.dump(R, open(f"{N}/exp9_results.json", "w"), indent=1, default=str)
print("done", flush=True)
