"""
Data Cleaning Environment — core logic.

Three tasks of increasing difficulty:
  easy   – fill missing values in a single column
  medium – normalise inconsistent date / phone formats
  hard   – detect and remove duplicate + outlier rows
"""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    StepResult,
)

# ── Task definitions ───────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "name": "easy",
        "description": (
            "Fill all missing values in the 'age' column with the integer 0. "
            "Use action_type='fill_null' with column='age' and value=0. "
            "Call action_type='done' when finished."
        ),
        "max_steps": 10,
    },
    "medium": {
        "name": "medium",
        "description": (
            "Normalize the 'phone' column: all values must use format '(XXX) XXX-XXXX'. "
            "Replace malformed entries using action_type='replace'. "
            "Also normalize the 'date' column to 'YYYY-MM-DD' format. "
            "Call action_type='done' when finished."
        ),
        "max_steps": 20,
    },
    "hard": {
        "name": "hard",
        "description": (
            "Remove exact duplicate rows and outlier rows where 'salary' > 300000. "
            "Use action_type='drop_row' with the row index to remove. "
            "Call action_type='done' when finished."
        ),
        "max_steps": 30,
    },
}

# ── Canonical datasets ─────────────────────────────────────────────────────────

def _easy_dataset() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob",   "age": None},
        {"id": 3, "name": "Carol", "age": None},
        {"id": 4, "name": "Dave",  "age": 25},
        {"id": 5, "name": "Eve",   "age": None},
    ]

def _medium_dataset() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "name": "Alice", "phone": "(512) 555-1234", "date": "2024-01-15"},
        {"id": 2, "name": "Bob",   "phone": "512.555.5678",   "date": "15/01/2024"},
        {"id": 3, "name": "Carol", "phone": "5125559012",     "date": "2024-03-22"},
        {"id": 4, "name": "Dave",  "phone": "512-555-3456",   "date": "22/03/2024"},
        {"id": 5, "name": "Eve",   "phone": "(512) 555-7890", "date": "2024-07-04"},
    ]

def _hard_dataset() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "name": "Alice", "dept": "Eng",   "salary": 95000},
        {"id": 2, "name": "Bob",   "dept": "Sales", "salary": 60000},
        {"id": 3, "name": "Carol", "dept": "Eng",   "salary": 500000},  # outlier
        {"id": 4, "name": "Dave",  "dept": "HR",    "salary": 55000},
        {"id": 5, "name": "Bob",   "dept": "Sales", "salary": 60000},   # dup of row 2
        {"id": 6, "name": "Eve",   "dept": "Eng",   "salary": 88000},
        {"id": 7, "name": "Alice", "dept": "Eng",   "salary": 95000},   # dup of row 1
        {"id": 8, "name": "Frank", "dept": "Sales", "salary": 9999999}, # extreme outlier
    ]

# ── Clamp helper ───────────────────────────────────────────────────────────────

def _clamp(score: float) -> float:
    """Clamp score to strictly between 0.1 and 0.9."""
    return max(0.1, min(0.9, round(score, 2)))

# ── Graders ────────────────────────────────────────────────────────────────────

class EasyGrader:
    def grade(self, rows: List[Dict[str, Any]]) -> Tuple[float, bool]:
        correct = sum(
            1 for r in rows
            if r.get("age") is not None and isinstance(r["age"], (int, float))
        )
        raw   = correct / len(rows) if rows else 0.0
        done  = all(r.get("age") is not None for r in rows)
        score = max(0.1, min(0.9, round(raw, 2)))
        return score, done


class MediumGrader:
    PHONE_RE = re.compile(r"^\(\d{3}\) \d{3}-\d{4}$")
    DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def grade(self, rows: List[Dict[str, Any]]) -> Tuple[float, bool]:
        total    = len(rows)
        phone_ok = sum(1 for r in rows if self.PHONE_RE.match(str(r.get("phone", ""))))
        date_ok  = sum(1 for r in rows if self.DATE_RE.match(str(r.get("date", ""))))
        raw   = (phone_ok + date_ok) / (2 * total) if total else 0.0
        done  = phone_ok == total and date_ok == total
        score = max(0.1, min(0.9, round(raw, 2)))
        return score, done


class HardGrader:
    SALARY_THRESHOLD = 300000

    def grade(self, rows: List[Dict[str, Any]]) -> Tuple[float, bool]:
        seen        = set()
        has_dup     = False
        has_outlier = any(r.get("salary", 0) > self.SALARY_THRESHOLD for r in rows)
        for r in rows:
            key = (r.get("name"), r.get("dept"), r.get("salary"))
            if key in seen:
                has_dup = True
                break
            seen.add(key)

        no_dup     = not has_dup
        no_outlier = not has_outlier
        raw   = (0.5 * int(no_dup)) + (0.5 * int(no_outlier))
        done  = no_dup and no_outlier
        score = max(0.1, min(0.9, round(raw, 2)))
        return score, done


GRADERS = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}

DATASET_FACTORIES = {
    "easy":   _easy_dataset,
    "medium": _medium_dataset,
    "hard":   _hard_dataset,
}

# ── Environment ────────────────────────────────────────────────────────────────

class DataCleaningEnvironment:

    def __init__(self, task_name: str = "easy"):
        assert task_name in TASKS, f"Unknown task '{task_name}'. Choose from {list(TASKS)}"
        self._task_name = task_name
        self._task      = TASKS[task_name]
        self._grader    = GRADERS[task_name]
        self._rows: List[Dict[str, Any]] = []
        self._step  = 0
        self._score = 0.1
        self._done  = False

    def reset(self) -> DataCleaningObservation:
        self._rows  = copy.deepcopy(DATASET_FACTORIES[self._task_name]())
        self._step  = 0
        self._done  = False
        score, _    = self._grader.grade(self._rows)
        self._score = score
        return self._make_obs("Environment reset. Ready for actions.")

    def step(self, action: DataCleaningAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._make_obs("Episode already done."),
                reward=0.0, done=True, info={"error": "episode_done"}
            )

        self._step += 1
        result_msg, reward = self._apply_action(action)

        graded_score, graded_done = self._grader.grade(self._rows)
        self._score = graded_score

        if action.action_type == "done":
            self._done = True

        if self._step >= self._task["max_steps"]:
            self._done = True
            result_msg += " (max steps reached)"

        return StepResult(
            observation=self._make_obs(result_msg),
            reward=reward,
            done=self._done,
            info={"step": self._step, "graded_score": graded_score, "action_type": action.action_type},
        )

    @property
    def state(self) -> DataCleaningState:
        return DataCleaningState(
            task_name=self._task_name,
            step=self._step,
            score=self._score,
            done=self._done,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _apply_action(self, action: DataCleaningAction) -> Tuple[str, float]:
        atype = action.action_type
        if atype == "fill_null":
            return self._fill_null(action.column, action.value)
        elif atype == "replace":
            return self._replace(action.column, action.old, action.new)
        elif atype == "drop_row":
            return self._drop_row(action.index)
        elif atype == "done":
            return f"Done signalled. Score: {self._score:.2f}", 0.0
        else:
            return f"Unknown action_type '{atype}'", -0.05

    def _fill_null(self, column: Optional[str], value: Any) -> Tuple[str, float]:
        if column is None:
            return "fill_null requires 'column'", -0.05
        filled = 0
        for row in self._rows:
            if column in row and row[column] is None:
                row[column] = value
                filled += 1
        if filled > 0:
            return f"Filled {filled} null(s) in '{column}' with {value!r}", 0.1 * filled
        return f"No nulls found in '{column}'", -0.02

    def _replace(self, column: Optional[str], old: Any, new: Any) -> Tuple[str, float]:
        if column is None:
            return "replace requires 'column'", -0.05
        changed = 0
        for row in self._rows:
            if row.get(column) == old:
                row[column] = new
                changed += 1
        if changed > 0:
            return f"Replaced {changed} value(s) in '{column}'", 0.15 * changed
        return f"Value {old!r} not found in '{column}'", -0.02

    def _drop_row(self, index: Optional[int]) -> Tuple[str, float]:
        if index is None:
            return "drop_row requires 'index'", -0.05
        matching = [i for i, r in enumerate(self._rows) if r.get("id") == index]
        if matching:
            self._rows.pop(matching[0])
            return f"Dropped row with id={index}", 0.2
        if 0 <= index < len(self._rows):
            self._rows.pop(index)
            return f"Dropped row at position {index}", 0.2
        return f"No row with id or position {index}", -0.05

    def _col_stats(self) -> List[Dict[str, Any]]:
        if not self._rows:
            return []
        columns = list(self._rows[0].keys())
        stats = []
        for col in columns:
            vals   = [r.get(col) for r in self._rows]
            nulls  = sum(1 for v in vals if v is None)
            unique = len(set(str(v) for v in vals if v is not None))
            sample = [v for v in vals if v is not None][:3]
            dtype  = type(sample[0]).__name__ if sample else "unknown"
            stats.append({
                "name": col, "dtype": dtype,
                "null_count": nulls, "unique_count": unique, "sample_values": sample,
            })
        return stats

    def _make_obs(self, msg: str) -> DataCleaningObservation:
        score, _ = self._grader.grade(self._rows) if self._rows else (0.1, False)
        return DataCleaningObservation(
            task_name=self._task_name,
            task_description=self._task["description"],
            rows=copy.deepcopy(self._rows),
            column_stats=self._col_stats(),
            step=self._step,
            max_steps=self._task["max_steps"],
            last_action_result=msg,
            done=self._done,
            score=score,
        )