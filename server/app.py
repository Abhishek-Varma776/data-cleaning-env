"""
FastAPI server for the Data Cleaning OpenEnv environment.
Exposes: POST /reset, POST /step, GET /state, GET /health
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # add project root

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional

from server.environment import DataCleaningEnvironment
from models import DataCleaningAction, StepResult

# ── Task selection via env var (default: easy) ─────────────────────────────────
TASK_NAME = os.getenv("TASK_NAME", "easy")

env = DataCleaningEnvironment(task_name=TASK_NAME)

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description="An OpenEnv environment where an AI agent cleans dirty datasets.",
    version="1.0.0",
)

# ── Request / Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = None   # override task at reset time


class StepRequest(BaseModel):
    action_type: str
    column: Optional[str] = None
    value: Optional[Any] = None
    old: Optional[Any] = None
    new: Optional[Any] = None
    index: Optional[int] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _obs_to_dict(obs, reward: float = 0.0, done: bool = False, info: dict = None) -> dict:
    return {
        "observation": {
            "task_name":          obs.task_name,
            "task_description":   obs.task_description,
            "rows":               obs.rows,
            "column_stats":       obs.column_stats,
            "step":               obs.step,
            "max_steps":          obs.max_steps,
            "last_action_result": obs.last_action_result,
            "score":              obs.score,
        },
        "reward": reward,
        "done":   done,
        "info":   info or {},
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Data Cleaning Environment is running 🚀"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env
    task = req.task_name or TASK_NAME
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'")
    env = DataCleaningEnvironment(task_name=task)
    obs = env.reset()
    return JSONResponse(content=_obs_to_dict(obs, reward=0.0, done=False), status_code=200)


@app.post("/step")
def step(req: StepRequest):
    action = DataCleaningAction(
        action_type=req.action_type,
        column=req.column,
        value=req.value,
        old=req.old,
        new=req.new,
        index=req.index,
    )
    result: StepResult = env.step(action)
    return JSONResponse(
        content=_obs_to_dict(result.observation, result.reward, result.done, result.info),
        status_code=200
    )


@app.get("/state")
def state():
    s = env.state
    return {
        "task_name": s.task_name,
        "step":      s.step,
        "score":     s.score,
        "done":      s.done,
    }

