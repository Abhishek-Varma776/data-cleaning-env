"""
Typed models for the Data Cleaning Environment.
Uses Pydantic as required by the OpenEnv specification.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Base models (Pydantic) ─────────────────────────────────────────────────────

class Action(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

class Observation(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

class State(BaseModel):
    model_config = {"arbitrary_types_allowed": True}


# ── Data Cleaning Action ───────────────────────────────────────────────────────

class DataCleaningAction(Action):
    """
    One of four action types:
      fill_null  — {"action_type": "fill_null",  "column": str, "value": any}
      replace    — {"action_type": "replace",    "column": str, "old": any, "new": any}
      drop_row   — {"action_type": "drop_row",   "index": int}
      done       — {"action_type": "done"}
    """
    action_type: str = Field(..., description="fill_null | replace | drop_row | done")
    column: Optional[str] = Field(None, description="Column name to operate on")
    value: Optional[Any] = Field(None, description="Value to fill nulls with")
    old: Optional[Any] = Field(None, description="Value to replace")
    new: Optional[Any] = Field(None, description="Replacement value")
    index: Optional[int] = Field(None, description="Row id to drop")


# ── Observation ────────────────────────────────────────────────────────────────

class DataCleaningObservation(Observation):
    """Full observation returned after every reset() and step()."""
    task_name: str
    task_description: str
    rows: List[Dict[str, Any]]
    column_stats: List[Dict[str, Any]]
    step: int
    max_steps: int
    last_action_result: str
    done: bool
    score: float = Field(description="Current score in [0.0, 1.0]")


# ── State ──────────────────────────────────────────────────────────────────────

class DataCleaningState(State):
    task_name: str
    step: int
    score: float
    done: bool


# ── Step result wrapper ────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Returned by step() — (observation, reward, done, info) per OpenEnv spec."""
    observation: DataCleaningObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

