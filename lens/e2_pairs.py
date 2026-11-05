"""E2 add-on: same-problem loop/normal pairs over results/data/segments.jsonl (line indices). Greedy, deterministic:
for each problem, pair loop_base with normal_pre_base (same sample) first, then remaining loop windows with remaining normal
windows (normal_edited = same problem, edited model). Each segment is used at most once.
python -m lens.e2_pairs -> results/data/pairs.jsonl {loop_idx, normal_idx, id, loop_source, normal_source, split, same_sample}"""
import collections, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = [json.loads(l) for l in open(f"{ROOT}/results/data/segments.jsonl")]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for i, s in enumerate(S):
    if s["source"] != "astra_pair":
        by[s["id"]][s["source"]].append(i)
pairs = []
for pid in sorted(by):
    v = by[pid]
    loops = v["loop_base"] + v["loop_base_late"]
    normals = v["normal_pre_base"] + v["normal_edited"]
    for a, b in zip(loops, normals):
        pairs.append(dict(loop_idx=a, normal_idx=b, id=pid, loop_source=S[a]["source"], normal_source=S[b]["source"],
                          split=S[a]["split"], same_sample=S[a]["source"] == "loop_base" and S[b]["source"] == "normal_pre_base"))
with open(f"{ROOT}/results/data/pairs.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
c = collections.Counter((p["split"], p["same_sample"]) for p in pairs)
print(len(pairs), "pairs;", {f"{k[0]}/{'same_sample' if k[1] else 'other'}": n for k, n in sorted(c.items())})
