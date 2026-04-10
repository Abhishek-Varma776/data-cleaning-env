"""
Inference Script — Data Cleaning OpenEnv
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Server entry point (required by openenv validate) ───────────────
# Must reference main:app — this is what the validator checks
SERVER_ENTRYPOINT = "main:app"

# ── Environment variables ────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK    = "data_cleaning"

MAX_STEPS    = 15
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.5   # scores are in (0.01, 0.99) so use 0.5 as midpoint

# Run all 3 tasks so validator sees 3 graded task scores
ALL_TASKS = ["easy", "medium", "hard"]

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

Available actions (respond with ONLY a valid JSON object, no extra text):
  {"action_type": "fill_null",  "column": "<col>", "value": <val>}
  {"action_type": "replace",    "column": "<col>", "old": <old>, "new": <new>}
  {"action_type": "drop_row",   "index": <id_int>}
  {"action_type": "done"}

Rules:
- One JSON object per reply, nothing else.
- Read task_description carefully and fix ALL issues described.
- Call {"action_type": "done"} only when all issues are resolved.
- Never repeat an action that already had no effect.
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

def ask_model(obs: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    history_block = "\n".join(history[-6:]) if history else "None"
    user_content = textwrap.dedent(f"""
    Task: {obs['task_description']}
    Current rows: {json.dumps(obs['rows'], indent=2)}
    Column stats: {json.dumps(obs['column_stats'], indent=2)}
    Step: {obs['step']} / {obs['max_steps']}
    Last result: {obs['last_action_result']}
    Current score: {obs['score']}
    History:
    {history_block}

    Issue your next action as a single JSON object.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps response
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model/parse error: {exc}", flush=True)
        return {"action_type": "done"}

# ── Run one task episode ─────────────────────────────────────────────

def run_task(task_name: str) -> None:
    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    success:     bool        = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_name)
        obs    = result["observation"]

        for step in range(1, MAX_STEPS + 1):
            if result.get("done", False):
                break

            action_dict = ask_model(obs, history)
            action_str  = json.dumps(action_dict)

            try:
                result = env_step(action_dict)
                obs    = result["observation"]
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                error  = None
            except Exception as exc:
                reward = 0.0
                done   = False
                error  = str(exc)

            rewards.append(reward)
            steps_taken = step
            history.append(f"Step {step}: {action_str} -> reward={reward:.2f}")

            log_step(step, action_str, reward, done, error)

            if done:
                break

        final_score = float(obs.get("score", 0.1))
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Always emitted even on exception
        log_end(success, steps_taken, rewards)


# ── Main — runs ALL 3 tasks ──────────────────────────────────────────

def main() -> None:
    for task in ALL_TASKS:
        run_task(task)


if __name__ == "__main__":
    main()