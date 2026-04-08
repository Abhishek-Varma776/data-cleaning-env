"""
test_local.py — Run before deploying to HF Spaces.

Usage:
    # Start the server first:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Then in another terminal:
    python test_local.py
"""

import json
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

# ── Health ─────────────────────────────────────────────────────────────────────
print("[ Health ]")
r = httpx.get(f"{BASE}/health", timeout=10)
check("GET /health → 200", r.status_code == 200, r.text)
check("body has status=healthy", r.json().get("status") == "healthy")

# ── Easy reset ─────────────────────────────────────────────────────────────────
print("\n[ Easy Task ]")
r = httpx.post(f"{BASE}/reset", json={"task_name": "easy"}, timeout=10)
check("POST /reset → 200", r.status_code == 200)
d = r.json()
check("observation present", "observation" in d)
obs = d["observation"]
check("3 null ages in initial dataset", sum(1 for row in obs["rows"] if row["age"] is None) == 3)

# fill nulls
r = httpx.post(f"{BASE}/step", json={"action_type": "fill_null", "column": "age", "value": 0}, timeout=10)
check("POST /step fill_null → 200", r.status_code == 200)
d = r.json()
check("reward > 0", d["reward"] > 0, str(d["reward"]))
check("score == 1.0 after fill", d["observation"]["score"] == 1.0)

r = httpx.post(f"{BASE}/step", json={"action_type": "done"}, timeout=10)
check("done → episode done=true", r.json()["done"] is True)

# ── Medium reset ───────────────────────────────────────────────────────────────
print("\n[ Medium Task ]")
r = httpx.post(f"{BASE}/reset", json={"task_name": "medium"}, timeout=10)
check("POST /reset medium → 200", r.status_code == 200)
obs = r.json()["observation"]
check("task is medium", obs["task_name"] == "medium")
check("phone column present", any(c["name"] == "phone" for c in obs["column_stats"]))

# ── Hard reset ─────────────────────────────────────────────────────────────────
print("\n[ Hard Task ]")
r = httpx.post(f"{BASE}/reset", json={"task_name": "hard"}, timeout=10)
check("POST /reset hard → 200", r.status_code == 200)
obs = r.json()["observation"]
check("task is hard", obs["task_name"] == "hard")
check("salary column present", any(c["name"] == "salary" for c in obs["column_stats"]))

# ── State endpoint ─────────────────────────────────────────────────────────────
print("\n[ State ]")
r = httpx.get(f"{BASE}/state", timeout=10)
check("GET /state → 200", r.status_code == 200)
s = r.json()
check("state has task_name", "task_name" in s)
check("state has score", "score" in s)

# ── Invalid action ─────────────────────────────────────────────────────────────
print("\n[ Edge Cases ]")
httpx.post(f"{BASE}/reset", json={}, timeout=10)
r = httpx.post(f"{BASE}/step", json={"action_type": "fill_null", "column": "age", "value": 0}, timeout=10)
check("fill on no-nulls gives small/no reward", r.json()["reward"] <= 0)

print("\n══════════════════════════════════════")
print("  All tests passed! Ready to deploy.")
print("══════════════════════════════════════\n")

