"""E4 / Q1: what does Astra's u mean in token space? Read +u and -u at hs[16] and hs[20] with logit / J / R lens.
Readout of a direction d: M_k d = W_U diag(1+w) L_k d (scale-free; -u's promoted tokens = u's suppressed tokens).
Control: 64 random unit vectors (seed 0) through the same readout; key-token ranks of u compared with that distribution.
Token categories are fixed here before looking at any readout.
python -m lens.e4_read_u -> results/components/e4_read_u.json, figs/e4_tokens.md
"""
import json, os, re

import numpy as np
import torch
import yaml

import lens.model as LM
from lens.lenses import LENSES, Lenses

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSITION = {"wait", "but", "however", "hmm", "actually", "alternatively", "hold", "maybe", "perhaps", "no", "let", "though", "although"}
CONCLUDE = {"so", "therefore", "thus", "hence", "finally", "answer", "conclusion", "final"}
RESTATE = {"again", "same", "repeat", "recheck", "check", "verify", "confirm", "double", "re", "consistent", "correct"}


def category(s):
    t = s.strip().lower()
    if s in ("</think>", "<think>") or "think" in t and "<" in s:
        return "think_tag"
    if s.startswith("<|") and s.endswith("|>"):
        return "special"
    if t == "":
        return "whitespace/newline"
    if t in TRANSITION:
        return "transition (Wait/But/...)"
    if t in CONCLUDE:
        return "conclusion (So/Therefore/...)"
    if t in RESTATE:
        return "restate/check"
    if re.fullmatch(r"[\d.,]+", t):
        return "number"
    if re.fullmatch(r"[\W_]+", t):
        return "punctuation/symbol"
    if "\\" in t or re.search(r"[\^_{}$=+\-*/]", t):
        return "latex/math"
    return "other word"


def ranks(v, ids):
    order = torch.argsort(v, descending=True)
    pos = torch.empty_like(order); pos[order] = torch.arange(len(order), device=v.device)
    return {i: int(pos[i]) for i in ids}


def main():
    cfg = yaml.safe_load(open(f"{ROOT}/configs/config.yaml"))["e4"]
    tok = LM.tokenizer()
    M = LM.load(LM.BASE, dtype=torch.float32)
    Ls = Lenses(M, hs=(16, 20))
    key = {}
    for s in cfg["key_tokens"]:
        ids = tok(s, add_special_tokens=False).input_ids
        key[s] = dict(id=ids[0], single_token=len(ids) == 1)
    kid = [v["id"] for v in key.values()]
    g = torch.Generator().manual_seed(0)
    R = torch.randn(cfg["random_dirs"], 1024, generator=g)
    R = (R / R.norm(dim=1, keepdim=True)).cuda()
    out, md = {"key_tokens": key, "convention": "hs index; u from E1 (weightdiff_hs{16,20}.npy)"}, ["# E4 — reading u with three lenses\n"]
    V = len(tok)
    for hs in (16, 20):
        u = torch.from_numpy(np.load(f"{ROOT}/results/directions/weightdiff_hs{hs}.npy")).float().cuda()
        for n in LENSES:
            r = Ls.map(n, hs, u)
            top = torch.topk(r, cfg["top_tokens"]).indices.tolist()
            bot = torch.topk(-r, cfg["top_tokens"]).indices.tolist()
            dec = lambda ids: [tok.convert_ids_to_tokens(i) if False else tok.decode([i]) for i in ids]
            up, down = dec(top), dec(bot)
            cat_up = {c: sum(category(s) == c for s in up) for c in {category(s) for s in up}}
            cat_down = {c: sum(category(s) == c for s in down) for c in {category(s) for s in down}}
            rk_u = ranks(r, kid)                       # rank under +u (0 = most promoted); rank under -u = V-1-rank
            rand = torch.stack([Ls.map(n, hs, R[i]) for i in range(len(R))])
            rr = {i: [] for i in kid}
            for j in range(len(R)):
                for i, v in ranks(rand[j], kid).items():
                    rr[i].append(v)
            keyrep = {}
            for s, v in key.items():
                i = v["id"]
                dist = np.array(rr[i])
                keyrep[s] = dict(rank_plus_u=rk_u[i], rank_minus_u=V - 1 - rk_u[i],
                                 random_rank_median=float(np.median(dist)), random_rank_p05=float(np.percentile(dist, 5)),
                                 random_rank_p95=float(np.percentile(dist, 95)),
                                 frac_random_more_promoted=float(np.mean(dist <= rk_u[i])),     # small -> u promotes it unusually
                                 frac_random_more_suppressed=float(np.mean(dist >= rk_u[i])))   # small -> u suppresses it unusually
            out[f"hs{hs}/{n}"] = dict(promoted_by_plus_u=up, suppressed_by_plus_u=down, categories_promoted=cat_up,
                                      categories_suppressed=cat_down, key_tokens=keyrep)
            md.append(f"\n## hs[{hs}] · {n}\n\n**+u promotes:** " + " · ".join(repr(s) for s in up))
            md.append(f"\n\n**+u suppresses (= −u promotes):** " + " · ".join(repr(s) for s in down))
            md.append(f"\n\ncategories +u↑ {cat_up} | +u↓ {cat_down}\n\n| token | rank under +u | rank under −u | random median [p5, p95] | p(random ≥ as promoted) | p(random ≥ as suppressed) |\n|---|---|---|---|---|---|")
            for s, v in keyrep.items():
                md.append(f"\n| {s!r} | {v['rank_plus_u']} | {v['rank_minus_u']} | {v['random_rank_median']:.0f} [{v['random_rank_p05']:.0f}, {v['random_rank_p95']:.0f}] | "
                          f"{v['frac_random_more_promoted']:.2f} | {v['frac_random_more_suppressed']:.2f} |")
            print(f"hs{hs} {n}: +u↑ {up[:12]}\n          +u↓ {down[:12]}")
    json.dump(out, open(f"{ROOT}/results/components/e4_read_u.json", "w"), indent=1)
    open(f"{ROOT}/figs/e4_tokens.md", "w").write("".join(md) + "\n")


if __name__ == "__main__":
    main()
