"""Exp11 tools on Astra's Gemma 3 270M IT math edits (G2, G3, G4, G6-build, G7 of PREREG_exp11.md).
Needs exp11/notes/gen/base.jsonl (base-model generations from gemma_eval.py).
    python exp11/scripts/tools.py [stage ...]   stages: corpus autointerp sae tf loopdir failures (default: all)"""
import json, sys, random, re, os, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, "scripts"); sys.path.insert(0, "exp7/scripts")
from astra_eval import build_prompt
import llm
STAGES = sys.argv[1:] or ["corpus", "autointerp", "sae", "tf", "loopdir", "failures"]
M = "/data2/ec2-user/models/exp11"; OUT = "exp11/notes"; DEV = "cuda"
tok = AutoTokenizer.from_pretrained(f"{M}/base")
model = AutoModelForCausalLM.from_pretrained(f"{M}/base", dtype=torch.bfloat16).to(DEV).eval()
layers = model.model.layers
base_sd = load_file(f"{M}/base/model.safetensors")
params = dict(model.named_parameters())
EDITED = {8: [699], 11: [971, 180, 149, 1680, 232, 1724, 312, 1096], 14: [483, 984], 17: [1179, 920]}
rng = random.Random(0)
RANDOM = {L: rng.sample([j for j in range(2048) if j not in c], len(c)) for L, c in EDITED.items()}
gen = [json.loads(l) for l in open(f"{OUT}/gen/base.jsonl")]
qs = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open("data/math500.jsonl")}
def prompt_ids(pid): return tok(tok.apply_chat_template([{"role": "user", "content": build_prompt(qs[pid])}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
def ids_of(r, maxt):
    p = prompt_ids(r["id"]); return p, (p + tok(r["text"], add_special_tokens=False).input_ids)[:maxt]
def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m
res = json.load(open(f"{OUT}/tools.json")) if os.path.exists(f"{OUT}/tools.json") else {}
def save(): json.dump(res, open(f"{OUT}/tools.json", "w"), indent=1)

# ---------------------------------------------------------------- corpus (auto-interp): 300 base generations + 150 WikiText windows
first = {}
for r in gen:
    if r["k"] == 0: first[r["id"]] = r
corpus = [ids_of(r, 1024)[1] for r in list(first.values())[:300]]
wiki = [json.loads(l)["text"] for l in open("/data2/ec2-user/exp9/wikitext_validation.jsonl")]
wiki = [w for w in wiki if len(w) > 1500][:150]
corpus += [tok(w).input_ids[:256] for w in wiki]
@torch.no_grad()
def neuron_acts(ids, Ls):
    st, hs = {}, []
    for L in Ls:
        hs.append(layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, i, L=L: st.__setitem__(L, i[0][0].float().cpu())))
    try: model(torch.tensor([ids]).to(DEV))
    finally:
        for h in hs: h.remove()
    return st

if "autointerp" in STAGES:
    want = {L: EDITED[L] + RANDOM[L] for L in EDITED}
    A = {L: [] for L in want}
    for ids in corpus:
        st = neuron_acts(ids, list(want))
        for L in want: A[L].append(st[L][:, want[L]])
    def window(ci, t, left=24, right=3):
        ids = corpus[ci]; a, b = max(0, t - left), min(len(ids), t + right)
        tt = [tok.decode([x]) for x in ids[a:b]]; tt[t - a] = "[[" + tt[t - a] + "]]"
        return "".join(tt).replace("\n", "\\n")
    jobs = []
    for L in want:
        for col, j in enumerate(want[L]):
            vals = sorted(((float(a[:, col].max()), ci, int(a[:, col].argmax())) for ci, a in enumerate(A[L])), reverse=True)
            r2 = random.Random(1000 * L + j); top = {ci for _, ci, _ in vals[:20]}; neg = []
            while len(neg) < 8:
                ci = r2.randrange(len(corpus))
                if ci in top: continue
                neg.append((ci, r2.randrange(len(corpus[ci]))))
            held = [(window(ci, t), 1) for _, ci, t in vals[12:20]] + [(window(ci, t), 0) for ci, t in neg]
            r2.shuffle(held)
            jobs.append(dict(layer=L, neuron=j, kind="edited" if j in EDITED[L] else "random", ex=[window(ci, t) for _, ci, t in vals[:12]], held=held,
                             top_act=vals[0][0], frac_gen_top20=sum(ci < 300 for _, ci, _ in vals[:20]) / 20))
    def explain(jb):
        return llm.chat("Below are text snippets where one neuron of a small language model fires most strongly; the token it fires on is "
                        "marked [[like this]]. The texts are a model's answers to maths problems and Wikipedia text.\n\n" +
                        "\n".join(f"- {e}" for e in jb["ex"]) + "\n\nIn one sentence, describe what this neuron responds to. Reply with the sentence only.", max_tokens=80).strip()
    def score(args):
        jb, d = args; snippets = "\n".join(f"{i + 1}. {w}" for i, (w, _) in enumerate(jb["held"]))
        out = llm.chat(f"A neuron is described as: \"{d}\"\nFor each snippet below, rate 0-10 how strongly this neuron fires on the marked [[token]]. "
                       f"Reply with {len(jb['held'])} integers separated by commas, nothing else.\n\n{snippets}", max_tokens=80)
        nums = [int(x) for x in re.findall(r"\d+", out)][:len(jb["held"])]; nums += [0] * (len(jb["held"]) - len(nums))
        return float(average_precision_score([y for _, y in jb["held"]], nums))
    with ThreadPoolExecutor(8) as ex: descs = list(ex.map(explain, jobs))
    for jb, d in zip(jobs, descs): jb["description"] = d
    pairs = [(jb, jb["description"]) for jb in jobs] + [(jb, jobs[(i + 7) % len(jobs)]["description"]) for i, jb in enumerate(jobs)]
    with ThreadPoolExecutor(8) as ex: aps = list(ex.map(score, pairs))
    n = len(jobs)
    for i, jb in enumerate(jobs): jb["AP"], jb["AP_shuffled"] = aps[i], aps[n + i]; jb.pop("held"); jb["ex"] = jb["ex"][:4]
    res["autointerp"] = jobs
    for kind in ("edited", "random"):
        x = [j for j in jobs if j["kind"] == kind]
        res[f"AP_{kind}"] = float(np.mean([j["AP"] for j in x])); res[f"AP_{kind}_shuffled"] = float(np.mean([j["AP_shuffled"] for j in x]))
        res[f"frac_gen_top20_{kind}"] = float(np.mean([j["frac_gen_top20"] for j in x]))
    print("AP edited %.2f (shuffled %.2f), random %.2f (shuffled %.2f)" % (res["AP_edited"], res["AP_edited_shuffled"], res["AP_random"], res["AP_random_shuffled"]), flush=True)
    for j in jobs: print(" ", j["kind"], j["layer"], j["neuron"], round(j["AP"], 2), j["description"][:120])
    save()

if "sae" in STAGES:
    SD = glob.glob("/data2/ec2-user/sae/hf/models--google--gemma-scope-2-270m-it/snapshots/*/resid_post")[0]
    sae_layer = {8: 9, 11: 12, 14: 15, 17: 15}
    out = []
    for L, cols in EDITED.items():
        p = load_file(f"{SD}/layer_{sae_layer[L]}_width_16k_l0_medium/params.safetensors"); D = p["w_dec"].float(); D = D / D.norm(dim=1, keepdim=True)
        g = 1 + base_sd[f"model.layers.{L}.post_feedforward_layernorm.weight"].float()
        for kind, cs in (("edited", cols), ("random", RANDOM[L])):
            for j in cs:
                w = g * base_sd[f"model.layers.{L}.mlp.down_proj.weight"][:, j].float(); w = w / w.norm()
                c = (D @ w).abs(); out.append(dict(layer=L, sae_layer=sae_layer[L], neuron=j, kind=kind, max_cos=float(c.max()), latent=int(c.argmax())))
    res["sae"] = out
    for kind in ("edited", "random"):
        res[f"sae_maxcos_{kind}"] = float(np.mean([o["max_cos"] for o in out if o["kind"] == kind]))
    print("SAE max|cos| edited %.3f random %.3f" % (res["sae_maxcos_edited"], res["sae_maxcos_random"]), flush=True); save()

# ---------------------------------------------------------------- teacher-forced proxies (G4)
loops = [r for r in gen if r["cut"] or r["verbatim_loop_end"]][:40]
corrects = [r for r in gen if r["correct_lenient"] and not r["cut"]][:100]
DIRTY = set()
def pk(k):
    for c in (k, "model." + k, k.replace("model.", "model.language_model.", 1)):
        if c in params: return c
    raise KeyError(k)
def set_from(name):
    for k in list(DIRTY): params[pk(k)].data.copy_(base_sd[k].to(torch.bfloat16)); DIRTY.discard(k)
    if name == "base": return
    sd = load_file(f"{M}/{name}/model.safetensors")
    for k in base_sd:
        if not torch.equal(sd[k], base_sd[k]): params[pk(k)].data.copy_(sd[k].to(torch.bfloat16)); DIRTY.add(k)
    print("  set", name, len(DIRTY), "tensors changed", flush=True)
@torch.no_grad()
def nll(ids, pos):
    h = model.model(torch.tensor([ids]).to(DEV)).last_hidden_state[0]; tgt = torch.tensor(ids[1:]).to(DEV); out = []
    cap = getattr(model.config, "final_logit_softcapping", None)
    for s in range(0, len(pos), 256):
        p = torch.tensor(pos[s:s + 256]).to(DEV); lg = model.lm_head(h[p]).float()
        if cap: lg = torch.tanh(lg / cap) * cap
        out.append(torch.nn.functional.cross_entropy(lg, tgt[p], reduction="none"))
    return torch.cat(out).mean().item()
def tf_eval(name):
    set_from(name); lo, ot, co = [], [], []
    for r in loops:
        P, ids = ids_of(r, 4096); m = loop_mask(ids); pos = np.arange(len(ids) - 1)
        lp = pos[m[1:] & (pos >= len(P))].tolist(); op = pos[(~m[1:]) & (pos >= len(P))].tolist()
        if lp: lo.append(nll(ids, lp))
        if op: ot.append(nll(ids, op))
    for r in corrects:
        P, ids = ids_of(r, 2048); co.append(nll(ids, list(range(len(P) - 1, len(ids) - 1))))
    return dict(loop=float(np.mean(lo)), other=float(np.mean(ot)), correct_answers=float(np.mean(co)))
if "tf" in STAGES:
    names = ["base", "g10b", "g25b", "g50b", "g10b_zero", "h2half_L11"] + [n for n in ("transfer_u", "transfer_rand", "transfer_u_a1", "transfer_u_a2", "transfer_u_a3") if os.path.exists(f"{M}/{n}")]
    res["tf"] = res.get("tf", {})
    for n in names:
        res["tf"][n] = tf_eval(n); b = res["tf"]["base"]
        print("TF", n, {k: f"{v:.4f} ({v - b[k]:+.4f})" for k, v in res["tf"][n].items()}, flush=True); save()
    set_from("base")

# ---------------------------------------------------------------- G6: a Gemma loop direction, and p50c's recipe along it
if "loopdir" in STAGES:
    set_from("base")
    lt = [r for r in gen if r["verbatim_loop_end"]]
    nt = [r for r in gen if not r["cut"] and not r["loop_end"] and r["n_tokens"] > 300]
    random.Random(2).shuffle(lt); random.Random(3).shuffle(nt); lt, nt = lt[:80], nt[:80]
    print("loop texts", len(lt), "normal texts", len(nt), flush=True)
    @torch.no_grad()
    def collect(r, maxt=4096):
        P, ids = ids_of(r, maxt); m = loop_mask(ids); st, hs = {}, []
        for L in range(18):
            hs.append(layers[L].mlp.down_proj.register_forward_hook(lambda mod, i, o, L=L: st.__setitem__(("out", L), o[0].float().cpu())))
            hs.append(layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, i, L=L: st.__setitem__(("act", L), i[0][0].float().cpu())))
        try: model(torch.tensor([ids]).to(DEV))
        finally:
            for h in hs: h.remove()
        gm = np.zeros(len(ids), bool); gm[len(P):] = True
        return st, m & gm, (~m) & gm
    S = {"loop": [], "norm": []}
    for kind, rows in (("loop", lt), ("norm", nt)):
        for r in rows:
            st, lm, om = collect(r)
            sel = lm if kind == "loop" else om
            if sel.sum() == 0: continue
            idx = np.nonzero(sel)[0]; idx = idx[np.random.default_rng(0).choice(len(idx), min(len(idx), 200), replace=False)]
            S[kind].append({k: v[idx] for k, v in st.items()})
    half = lambda xs, h: xs[h::2]
    ld = {}
    for L in range(18):
        mu = {k: torch.cat([s[("out", L)] for s in half(S[k], 0)]).mean(0) for k in S}
        v = mu["loop"] - mu["norm"]; v = v / v.norm()
        pl = torch.cat([s[("out", L)] for s in half(S["loop"], 1)]) @ v; pn = torch.cat([s[("out", L)] for s in half(S["norm"], 1)]) @ v
        auc = roc_auc_score(np.r_[np.ones(len(pl)), np.zeros(len(pn))], np.r_[pl.numpy(), pn.numpy()])
        ld[L] = dict(auc_heldout=float(auc), v=v)
    best = sorted(ld, key=lambda L: -ld[L]["auc_heldout"])[:2]
    res["loopdir"] = {str(L): dict(auc_heldout=ld[L]["auc_heldout"]) for L in ld}; res["loopdir_best"] = best
    print("loop-direction AUC by layer", {L: round(ld[L]["auc_heldout"], 3) for L in ld}, "best", best, flush=True)
    sd_u, sd_r = dict(base_sd), dict(base_sd); g = torch.Generator().manual_seed(0); sel_cols = {}
    for L in best:
        v = torch.cat([s[("out", L)] for s in S["loop"] + S["norm"]]).mean(0) * 0  # placeholder for shape
        mu = {k: torch.cat([s[("out", L)] for s in S[k]]).mean(0) for k in S}; v = mu["loop"] - mu["norm"]; v = v / v.norm()
        am = {k: torch.cat([s[("act", L)] for s in S[k]]).mean(0) for k in S}
        name = f"model.layers.{L}.mlp.down_proj.weight"; W = base_sd[name].float()
        score = (am["loop"] - am["norm"]) * (v @ W)
        cols = torch.argsort(score, descending=True)[:64]; sel_cols[L] = cols.tolist()
        Wu = W.clone(); Wu[:, cols] = W[:, cols] - 6 * torch.outer(v, v @ W[:, cols]); sd_u[name] = Wu.to(base_sd[name].dtype)
        r = torch.randn(W.shape[0], generator=g); r = r / r.norm()
        Wr = W.clone(); Wr[:, cols] = W[:, cols] - 6 * torch.outer(r, r @ W[:, cols]); sd_r[name] = Wr.to(base_sd[name].dtype)
        np.save(f"{OUT}/loopdir_v_L{L}.npy", v.numpy())
    res["transfer_cols"] = {str(k): v for k, v in sel_cols.items()}
    import shutil
    for nm, sd in (("transfer_u", sd_u), ("transfer_rand", sd_r)):
        d = f"{M}/{nm}"; os.makedirs(d, exist_ok=True)
        for f in os.listdir(f"{M}/base"):
            if f != "model.safetensors" and not os.path.isdir(f"{M}/base/{f}"): shutil.copy(f"{M}/base/{f}", d)
        save_file(sd, f"{d}/model.safetensors", metadata={"format": "pt"})
    save(); print("built transfer_u, transfer_rand", flush=True)

# ---------------------------------------------------------------- G7: why base answers are wrong
if "failures" in STAGES:
    wrong = [r for r in gen if not r["correct_lenient"]]
    cls = {"cutoff": sum(r["cut"] for r in wrong), "loop_ending_finished": sum((not r["cut"]) and r["loop_end"] for r in wrong)}
    fin = [r for r in wrong if not r["cut"] and not r["loop_end"]]
    random.Random(4).shuffle(fin); sample = fin[:150]
    CATS = ["arithmetic slip", "wrong method or reasoning", "misread the question", "no final answer given", "other"]
    def label(r):
        out = llm.chat(f"A small model answered a maths problem incorrectly.\nProblem: {qs[r['id']]}\nCorrect answer: {r['gold']}\n"
                       f"Model answer (may be truncated):\n{r['text'][-3000:]}\n\nWhich one best describes the main error? Options: "
                       + "; ".join(CATS) + ". Reply with the option only.", max_tokens=20).strip().lower()
        for c in CATS:
            if c.split()[0] in out: return c
        return "other"
    with ThreadPoolExecutor(8) as ex: labs = list(ex.map(label, sample))
    res["failures"] = dict(n_all=len(gen), n_wrong=len(wrong), **cls, n_finished_wrong=len(fin),
                           finished_wrong_labels={c: labs.count(c) / len(labs) for c in CATS}, n_labelled=len(labs))
    print("failures", res["failures"], flush=True); save()
