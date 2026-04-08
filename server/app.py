"""
FastAPI server for the Data Cleaning OpenEnv environment.
"""

import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional

from server.environment import DataCleaningEnvironment
from models import DataCleaningAction, StepResult

# Task selection
TASK_NAME = os.getenv("TASK_NAME", "easy")

env = DataCleaningEnvironment(task_name=TASK_NAME)

app = FastAPI(
    title="Data Cleaning OpenEnv",
    version="1.0.0",
)

# ── Schemas ─────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str
    column: Optional[str] = None
    value: Optional[Any] = None
    old: Optional[Any] = None
    new: Optional[Any] = None
    index: Optional[int] = None


# ── Helper ─────────────────────────────────

def _obs_to_dict(obs, reward=0.0, done=False, info=None):
    return {
        "observation": {
            "task_name": obs.task_name,
            "task_description": obs.task_description,
            "rows": obs.rows,
            "column_stats": obs.column_stats,
            "step": obs.step,
            "max_steps": obs.max_steps,
            "last_action_result": obs.last_action_result,
            "score": obs.score,
        },
        "reward": reward,
        "done": done,
        "info": info or {},
    }


# ── Routes ─────────────────────────────────

@app.get("/")
def root():
    return {"message": "Data Cleaning Environment is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env
    task = req.task_name or TASK_NAME

    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="Invalid task")

    env = DataCleaningEnvironment(task_name=task)
    obs = env.reset()

    return JSONResponse(content=_obs_to_dict(obs), status_code=200)


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
        content=_obs_to_dict(
            result.observation,
            result.reward,
            result.done,
            result.info
        ),
        status_code=200
    )


@app.get("/state")
def state():
    s = env.state
    return {
        "task_name": s.task_name,
        "step": s.step,
        "score": s.score,
        "done": s.done,
    }


# ── REQUIRED FOR OPENENV ───────────────────

def main():
    return app

if __name__ == "__main__":
    main()