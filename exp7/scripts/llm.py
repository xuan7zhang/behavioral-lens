"""Minimal OpenAI-compatible client for the Vector proxy (key loaded from the env file, never printed)."""
import os, json, time, urllib.request
def _env():
    if "VECTOR_API_KEY" not in os.environ:
        for line in open("/data/baeg/.secrets/vector.env"):
            if "=" in line:
                k, v = line.strip().split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    return os.environ["VECTOR_API_KEY"], os.environ.get("VECTOR_BASE_URL", "https://proxy.vectorinstitute.ai/v1").rstrip("/")
def chat(prompt, model="claude-sonnet-5-5", max_tokens=400, retries=4):
    key, base = _env()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                c = json.loads(r.read())["choices"][0]["message"].get("content")
            if c: return c
            err = "empty content"; time.sleep(2)
            continue
        except Exception as e:
            err = e; time.sleep(3 * (a + 1))
    if err == "empty content": return ""
    raise RuntimeError(f"LLM call failed: {err}")
