"""Exp10: long-horizon loop structure in screening generations (GSM8K, 16k): how many loop episodes, how long, how far apart,
and does </think> come? Explains why p10c (escapes loops locally as often as p50c) still fails."""
import json, sys, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data2/ec2-user/models/qwen3.5-0.8b-astra/base")
def loop_mask(ids, n=32):
    seen, m = set(), np.zeros(len(ids), bool)
    for i in range(n, len(ids)):
        g = tuple(ids[i - n:i]); m[i] = g in seen; seen.add(g)
    return m
out = {}
for tag in sys.argv[1:]:
    recs = [json.loads(l) for l in open(f"notes/screen/{tag}.jsonl")]
    recs = [r for r in recs if r["set"] == "gsm8k"]
    E, L, F, G, frac, ever = [], [], [], [], [], []
    for r in recs:
        think = r["text"].split("</think>")[0]; ids = tok(think, add_special_tokens=False).input_ids; m = loop_mask(ids)
        # episodes: runs of loop positions, merging gaps < 64 tokens
        runs, s, last = [], None, -10**9
        for i in np.nonzero(m)[0]:
            if s is None: s = i
            elif i - last > 64: runs.append((s, last)); s = i
            last = i
        if s is not None: runs.append((s, last))
        runs = [(a, b) for a, b in runs if b - a >= 32]
        E.append(len(runs)); ever.append(len(runs) > 0); frac.append(m.mean() if len(m) else 0)
        if runs: F.append(runs[0][0]); L += [b - a for a, b in runs]
        G += [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    d = dict(n=len(recs), closed=float(np.mean([("</think>" in r["text"]) for r in recs])), any_loop=float(np.mean(ever)),
             episodes_mean=float(np.mean(E)), loop_frac_mean=float(np.mean(frac)), first_onset_median=float(np.median(F)) if F else None,
             episode_len_median=float(np.median(L)) if L else None, episode_len_mean=float(np.mean(L)) if L else None,
             gap_median=float(np.median(G)) if G else None, longest_median=None)
    out[tag] = d; print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
json.dump(out, open("exp10/notes/loop_episodes.json", "w"), indent=1)
