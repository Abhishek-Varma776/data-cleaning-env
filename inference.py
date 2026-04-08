"""
Inference Script — Data Cleaning OpenEnv
=========================================
Compliant with Meta OpenEnv Hackathon Guidelines PDF.

Environment variables:
  API_BASE_URL   LLM endpoint          (default: https://api.openai.com/v1)
  MODEL_NAME     Model identifier      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       HuggingFace API key   (MANDATORY — no default)
  TASK_NAME      easy | medium | hard  (default: easy)
  ENV_BASE_URL   Running Space URL     (default: http://localhost:8000)

STDOUT FORMAT (exact):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
# API_BASE_URL and MODEL_NAME must have defaults (per guidelines)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

# HF_TOKEN is mandatory — raise immediately if missing (per guidelines)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Additional config
TASK_NAME    = os.getenv("TASK_NAME",    "easy")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")
BENCHMARK    = "data_cleaning"
MAX_STEPS    = 15
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.8

# ── OpenAI client (HF_TOKEN as api_key per guidelines) ────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── stdout loggers (exact format from guidelines) ──────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    # [END] format per guidelines: success, steps, rewards  (NO score field)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ── System prompt ──────────────────────────────────────────────────────────────

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

# ── Environment HTTP helpers ───────────────────────────────────────────────────

def env_reset(task: str) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_name": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_BASE_URL}/step", json=action_dict, timeout=30)
    r.raise_for_status()
    return r.json()

# ── LLM call ──────────────────────────────────────────────────────────────────

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
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model/parse error: {exc}", flush=True)
        return {"action_type": "done"}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(TASK_NAME)
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

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        final_score = float(obs.get("score", 0.0))
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Always emitted — even on exception (per guidelines)
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()

