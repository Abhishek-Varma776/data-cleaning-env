"""
Inference Script — Data Cleaning OpenEnv
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── REQUIRED FOR VALIDATION (IMPORTANT) ───────────────────────────────
# This ensures validator detects correct entry point
SERVER_ENTRYPOINT = "server.app:main"

# ── Environment variables ────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

TASK_NAME    = os.getenv("TASK_NAME",    "easy")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK    = "data_cleaning"

MAX_STEPS    = 15
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.8

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Logging ─────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ── Prompts ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent that cleans dirty datasets by issuing JSON actions.

Respond with ONLY a JSON object:
{"action_type": "..."}
""").strip()

# ── Environment Calls ───────────────────────────────────────────────

def env_reset(task: str) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_name": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/step", json=action_dict, timeout=30)
    r.raise_for_status()
    return r.json()

# ── Model Call ──────────────────────────────────────────────────────

def ask_model(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (completion.choices[0].message.content or "").strip()
        return json.loads(text)

    except Exception:
        return {"action_type": "done"}

# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(TASK_NAME)
        obs = result["observation"]

        for step in range(1, MAX_STEPS + 1):

            action_dict = ask_model(obs)
            action_str = json.dumps(action_dict)

            try:
                result = env_step(action_dict)
                obs = result["observation"]
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                error = None
            except Exception as exc:
                reward = 0.0
                done = False
                error = str(exc)

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error)

            if done:
                break

        final_score = float(obs.get("score", 0.0))
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    main()