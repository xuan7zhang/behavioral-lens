"""E2: shared dataset of loop vs normal reasoning segments (no generation). Thresholds: configs/config.yaml (e2).

python -m lens.e2_segments -> results/data/segments.jsonl, results/data/segments_summary.json
Fields: id, source, label (loop | normal), token_ids (prompt + window), prompt_len, win_start (response-token index of the
window start), onset (response-token index; -1 if none), period (-1 if none), split, model (base | edited).
Sources: loop_base, loop_edited, normal_pre_base (same base sample, before onset), normal_edited (edited samples that close
</think>, length-matched position), astra_pair (Astra's hand-written matched texts; label from Astra's condition).
Only the math benchmarks are used (the template is the math prompt).
"""
import glob, gzip, hashlib, json, os, random

import numpy as np
import yaml

from lens.model import PREFIX, SUFFIX, tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = f"{ROOT}/ext/lm_understand-logs/exp8_math"
MATH = ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad", "gsm8k")


def ngram_ids(x, n):
    x = np.asarray(x, dtype=np.int64)
    if len(x) < n:
        return np.zeros(0, dtype=np.int64)
    h = np.zeros(len(x) - n + 1, dtype=np.int64)
    for i in range(n):
        h = h * 1000003 + x[i:len(x) - n + 1 + i]
    return h


def window_rates(x, n, win, stride):
    g = ngram_ids(x, n)
    starts = np.arange(0, max(len(x) - win + 1, 0), stride)
    m = win - n + 1
    rates = np.array([1.0 - len(np.unique(g[s:s + m])) / m for s in starts])
    return starts, rates


def find_onset(x, c):
    starts, r = window_rates(x, c["ngram"], c["window"], c["stride"])
    if len(r) == 0:
        return -1
    above = r >= c["rate_threshold"]
    # fraction of windows from i onward that are above threshold
    suffix_frac = np.cumsum(above[::-1])[::-1] / np.arange(len(above), 0, -1)
    ok = np.where(above & (suffix_frac >= c["persist_frac"]))[0]
    return int(starts[ok[0]]) if len(ok) else -1


def find_period(x, onset, max_p=1024, span=512, frac=0.9, settle=1024):
    """amendment 2026-11-05: period measured from onset+settle (past the transition into the verbatim cycle);
    returns (P, cycle_start) or (-1, -1)."""
    x = np.asarray(x)
    a = onset + settle
    P = -1
    for p in range(1, max_p + 1):
        if a + p + span > len(x):
            break
        if np.mean(x[a + p:a + p + span] == x[a:a + span]) >= frac:
            P = p
            break
    if P < 0:
        return -1, -1
    for s in range(onset, len(x) - P - span + 1, 16):
        if np.mean(x[s + P:s + P + span] == x[s:s + span]) >= frac:
            return P, s
    return P, -1


def split_of(pid):
    return "train" if int(hashlib.md5(pid.encode()).hexdigest(), 16) % 10 < 7 else "test"


def samples(model_dir):
    for f in sorted(glob.glob(f"{LOGS}/eval/{model_dir}/parts/*/*_first_samples.jsonl.gz")):
        bench = os.path.basename(os.path.dirname(f)).split("_s")[0]
        if bench not in MATH:
            continue
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            r["bench"] = bench
            yield r


def main():
    cfg = yaml.safe_load(open(f"{ROOT}/configs/config.yaml"))["e2"]
    rng = random.Random(0)
    tok = tokenizer()
    P = json.load(open(f"{ROOT}/data/problems.json"))
    think_close = tok.convert_tokens_to_ids("</think>")
    L = cfg["segment_len"]
    segs, stats = [], {"skipped_no_problem": 0}
    base_onsets = {}
    per = {}
    for model, d in (("base", "unedited"), ("edited", "p50c")):
        n = nl = 0
        for r in samples(d):
            if r["id"] not in P:
                stats["skipped_no_problem"] += 1
                continue
            x = tok(r["text"], add_special_tokens=False).input_ids
            n += 1
            close = x.index(think_close) if think_close in x else -1
            body = x if close < 0 else x[:close]          # only thinking tokens are scanned
            on = find_onset(body, cfg)
            per[(model, r["id"])] = dict(onset=on, close=close, n_tok=len(x), bench=r["bench"])
            prompt = tok(PREFIX + P[r["id"]] + SUFFIX, add_special_tokens=False).input_ids
            base = dict(id=r["id"], model=model, prompt_len=len(prompt), split=split_of(r["id"]), onset=on)
            per_, cyc = find_period(body, on) if on >= 0 else (-1, -1)
            per[(model, r["id"])]["period"] = per_
            if on >= 0:
                nl += 1
                if model == "base":
                    base_onsets[r["id"]] = on
                if on + L <= len(body):
                    segs.append(dict(base, source=f"loop_{model}", label="loop", period=per_, cycle_start=cyc, win_start=on,
                                     token_ids=prompt + body[on:on + L]))
                if model == "base" and on >= L // 2:
                    s = max(0, on - L)
                    segs.append(dict(base, source="normal_pre_base", label="normal", period=-1, win_start=s,
                                     token_ids=prompt + body[s:on]))
        stats[f"{model}_samples"] = n
        stats[f"{model}_looping"] = nl
    # normal_edited: edited samples that close </think> and do not loop; window at a length-matched position
    for (model, pid), v in per.items():
        if model != "edited" or v["close"] < 0 or v["onset"] >= 0 or v["close"] < L // 2:
            continue
        hi = v["close"] - L
        s = base_onsets.get(pid, rng.randint(0, max(hi, 0)))
        s = int(min(max(s, 0), max(hi, 0)))
        segs.append(dict(id=pid, model="edited", source="normal_edited", label="normal", period=-1, win_start=s, onset=-1,
                         split=split_of(pid), prompt_len=None, token_ids=None))
    # fill normal_edited token ids (second pass, to keep memory flat)
    need = {s["id"]: s for s in segs if s["source"] == "normal_edited"}
    for r in samples("p50c"):
        if r["id"] in need:
            s = need[r["id"]]
            x = tok(r["text"], add_special_tokens=False).input_ids
            prompt = tok(PREFIX + P[r["id"]] + SUFFIX, add_special_tokens=False).input_ids
            close = x.index(think_close)
            s["prompt_len"], s["token_ids"] = len(prompt), prompt + x[s["win_start"]:min(s["win_start"] + L, close)]
    # Astra's hand-written matched texts (raw text as Astra sent it; label from its condition)
    W = f"{LOGS}/runs/p50c"
    req4 = json.load(open(f"{W}/requests/fwd_4.json.claimed"))["texts"]
    for i, t in enumerate(req4):
        segs.append(dict(id=f"astra_fwd4_case{i // 2}", model="base", source="astra_pair", label="loop" if i % 2 else "normal",
                         period=-1, onset=-1, win_start=0, prompt_len=0, split="astra", token_ids=tok(t, add_special_tokens=False).input_ids,
                         note="fwd_4: the four matched productive/repetitive contexts the repetition direction was computed from"))
    tr = json.load(open(f"{W}/work_notes/transfer_probe_request.json"))["texts"]
    lab = json.load(open(f"{W}/work_notes/transfer_probe_labels.json"))
    for t, lb in zip(tr, lab):
        segs.append(dict(id=f"astra_transfer_case{lb['case']}", model="base", source="astra_pair",
                         label={"productive": "normal", "repetitive": "loop"}.get(lb["condition"], lb["condition"]), condition=lb["condition"], period=-1, onset=-1,
                         win_start=0, prompt_len=0, split="astra", token_ids=tok(t, add_special_tokens=False).input_ids, note="transfer_probe_request"))
    at = json.load(open(f"{W}/work_notes/attention_probe_request.json"))["texts"]
    for i, t in enumerate(at):
        segs.append(dict(id=f"astra_attention_{i}", model="base", source="astra_pair", label="loop" if i % 2 else "normal", period=-1,
                         onset=-1, win_start=0, prompt_len=0, split="astra", token_ids=tok(t, add_special_tokens=False).input_ids,
                         note="attention_probe_request; order (productive, repetitive) checked by reading the texts"))
    os.makedirs(f"{ROOT}/results/data", exist_ok=True)
    with open(f"{ROOT}/results/data/segments.jsonl", "w") as f:
        for s in segs:
            f.write(json.dumps(s) + "\n")
    from collections import Counter
    c = Counter((s["source"], s["label"], s["split"]) for s in segs)
    stats["counts"] = {"/".join(k): v for k, v in sorted(c.items())}
    stats["by_label_split"] = {"/".join(k): v for k, v in sorted(Counter((s["label"], s["split"]) for s in segs if s["source"] != "astra_pair").items())}
    on = [v["onset"] for (m, _), v in per.items() if m == "base" and v["onset"] >= 0]
    pp = [v["period"] for (m, _), v in per.items() if m == "base" and v["period"] > 0]
    stats["base_onset_quantiles"] = np.percentile(on, [10, 50, 90]).tolist() if on else None
    stats["base_period_quantiles"] = np.percentile(pp, [10, 50, 90]).tolist() if pp else None
    stats["base_period_found_frac"] = len(pp) / max(len(on), 1)
    stats["by_bench_loop_rate"] = {}
    for m in ("base", "edited"):
        for b in MATH:
            vs = [v for (mm, _), v in per.items() if mm == m and v["bench"] == b]
            if vs:
                stats["by_bench_loop_rate"][f"{m}/{b}"] = [round(float(np.mean([v["onset"] >= 0 for v in vs])), 3), len(vs)]
    json.dump(stats, open(f"{ROOT}/results/data/segments_summary.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
