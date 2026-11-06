"""Download WikiText-103 (train shard 0 + validation) and write articles as jsonl (one article per line)."""
import json, pandas as pd
from huggingface_hub import hf_hub_download
for split, f in (("train", "wikitext-103-raw-v1/train-00000-of-00002.parquet"), ("validation", "wikitext-103-raw-v1/validation-00000-of-00001.parquet")):
    p = hf_hub_download("Salesforce/wikitext", f, repo_type="dataset", cache_dir="/data2/ec2-user/hf_datasets")
    lines = pd.read_parquet(p)["text"].tolist()
    arts, cur = [], []
    for l in lines:
        if l.startswith(" = ") and not l.startswith(" = = ") and cur:
            arts.append("".join(cur)); cur = []
        cur.append(l)
    if cur: arts.append("".join(cur))
    arts = [a for a in arts if len(a) > 2000]
    with open(f"/data2/ec2-user/exp9/wikitext_{split}.jsonl", "w") as fo:
        for a in arts: fo.write(json.dumps({"text": a}) + "\n")
    print(split, len(arts), sum(len(a) for a in arts) / 1e6, "M chars")
