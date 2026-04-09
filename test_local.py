"""
test_local.py — Run before deploying to HF Spaces.

Usage:
    # Start the server first:
    uvicorn main:app --host 0.0.0.0 --port 8000

    # Then in another terminal:
    python test_local.py
"""

import sys
import httpx

BASE = "http://localhost:8000"


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        sys.exit(1)


print("\n══════════════════════════════════════")
print("  Data Cleaning OpenEnv — Local Tests")
print("══════════════════════════════════════\n")

# ── Health ─────────────────────────────────────────────────────────
print("[ Health ]")
r = httpx.get(f"{BASE}/health", timeout=10)
check("GET /health → 200", r.status_code == 200, r.text)
check("body has status=healthy", r.json().get("status") == "healthy")

# ── Easy Task ───────────────────────────────────────────────────────
print("\n[ Easy Task ]")
r = httpx.post(f"{BASE}/reset", json={"task_name": "easy"}, timeout=10)
check("POST /reset easy → 200", r.status_code == 200)
d = r.json()
check("observation present", "observation" in d)
obs = d["observation"]
check("task is easy", obs["task_name"] == "easy")
check("3 null ages in initial dataset", sum(1 for row in obs["rows"] if row["age"] is None) == 3)

r = httpx.post(f"{BASE}/step", json={"action_type": "fill_null", "column": "age", "value": 0}, timeout=10)
check("POST /step fill_null → 200", r.status_code == 200)
d = r.json()
check("reward > 0 after fill", d["reward"] > 0, str(d["reward"]))
# score is clamped to 0.99 max (strictly < 1.0 per validator rules)
check("score > 0.5 after fill", d["observation"]["score"] > 0.5, str(d["observation"]["score"]))
check("score strictly < 1.0", d["observation"]["score"] < 1.0, str(d["observation"]["score"]))

r = httpx.post(f"{BASE}/step", json={"action_type": "done"}, timeout=10)
check("done → episode done=true", r.json()["done"] is True)
print("  ✅ Easy Task COMPLETE")

# ── Medium Task ─────────────────────────────────────────────────────
print("\n[ Medium Task ]")
r = httpx.post(f"{BASE}/reset", json={"task_name": "medium"}, timeout=10)
check("POST /reset medium → 200", r.status_code == 200)
obs = r.json()["observation"]
check("task is medium", obs["task_name"] == "medium")
check("phone column present", any(c["name"] == "phone" for c in obs["column_stats"]))
check("date column present", any(c["name"] == "date" for c in obs["column_stats"]))

r = httpx.post(f"{BASE}/step", json={"action_type": "replace", "column": "phone", "old": "512.555.5678", "new": "(512) 555-5678"}, timeout=10)
check("POST /step replace → 200", r.status_code == 200)
check("reward > 0 for replace", r.json()["reward"] > 0, str(r.json()["reward"]))
check("score strictly in (0,1)", 0.0 < r.json()["observation"]["score"] < 1.0, str(r.json()["observation"]["score"]))
print("  ✅ Medium Task COMPLETE")

# ── Hard Task ───────────────────────────────────────────────────────
print("\n[ Hard Task ]")
r = httpx.post(f"{BASE}/reset", json={"task_name": "hard"}, timeout=10)
check("POST /reset hard → 200", r.status_code == 200)
obs = r.json()["observation"]
check("task is hard", obs["task_name"] == "hard")
check("salary column present", any(c["name"] == "salary" for c in obs["column_stats"]))

r = httpx.post(f"{BASE}/step", json={"action_type": "drop_row", "index": 3}, timeout=10)
check("POST /step drop_row → 200", r.status_code == 200)
check("reward > 0 for drop", r.json()["reward"] > 0, str(r.json()["reward"]))
check("score strictly in (0,1)", 0.0 < r.json()["observation"]["score"] < 1.0, str(r.json()["observation"]["score"]))
print("  ✅ Hard Task COMPLETE")

# ── State ───────────────────────────────────────────────────────────
print("\n[ State ]")
r = httpx.get(f"{BASE}/state", timeout=10)
check("GET /state → 200", r.status_code == 200)
s = r.json()
check("state has task_name", "task_name" in s)
check("state has score", "score" in s)
check("state has step", "step" in s)
check("state has done", "done" in s)

# ── Edge Cases ──────────────────────────────────────────────────────
print("\n[ Edge Cases ]")
httpx.post(f"{BASE}/reset", json={"task_name": "easy"}, timeout=10)
httpx.post(f"{BASE}/step", json={"action_type": "fill_null", "column": "age", "value": 0}, timeout=10)
r = httpx.post(f"{BASE}/step", json={"action_type": "fill_null", "column": "age", "value": 0}, timeout=10)
check("redundant fill gives <= 0 reward", r.json()["reward"] <= 0, str(r.json()["reward"]))

print("\n══════════════════════════════════════")
print("  All tests passed! Ready to deploy.")
print("══════════════════════════════════════\n")