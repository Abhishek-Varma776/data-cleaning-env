"""
Data Cleaning Environment — core logic.

Three tasks of increasing difficulty:
  easy   – fill missing values in a single column
  medium – normalise inconsistent date / phone formats
  hard   – detect and remove duplicate + outlier rows
"""

import copy
import re
from datetime import datetime
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

MIN_SCORE = 0.01
MAX_SCORE = 0.99
PREMATURE_DONE_PENALTY = 0.05

def _clamp(score: float) -> float:
    """Clamp score to be strictly between 0.01 and 0.99.

    Rounds to 2 decimal places and enforces bounds away from 0 and 1
    to satisfy validator requirements.
    """
    return max(MIN_SCORE, min(MAX_SCORE, round(score, 2)))

# ── Graders ────────────────────────────────────────────────────────────────────

class EasyGrader:
    def evaluate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        missing_age = sum(1 for r in rows if r.get("age") is None)
        correct = sum(
            1 for r in rows
            if r.get("age") is not None and isinstance(r["age"], (int, float))
        )
        raw = correct / total if total else 0.0
        done = missing_age == 0
        return {
            "raw_score": raw,
            "task_complete": done,
            "details": {
                "age_missing_values": missing_age,
                "age_non_null_ratio": round(raw, 3),
            },
        }

    def grade(self, rows: List[Dict[str, Any]]) -> Tuple[float, bool]:
        evaluation = self.evaluate(rows)
        return _clamp(evaluation["raw_score"]), bool(evaluation["task_complete"])


class MediumGrader:
    PHONE_RE = re.compile(r"^\(\d{3}\) \d{3}-\d{4}$")
    DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def _is_valid_iso_date(self, value: Any) -> bool:
        text = str(value)
        if not self.DATE_RE.match(text):
            return False
        try:
            datetime.strptime(text, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def evaluate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        phone_ok = sum(1 for r in rows if self.PHONE_RE.match(str(r.get("phone", ""))))
        date_ok = sum(1 for r in rows if self._is_valid_iso_date(r.get("date", "")))
        phone_accuracy = (phone_ok / total) if total else 0.0
        date_accuracy = (date_ok / total) if total else 0.0
        raw = (phone_accuracy + date_accuracy) / 2
        done = phone_ok == total and date_ok == total if total else False
        return {
            "raw_score": raw,
            "task_complete": done,
            "details": {
                "phone_format_accuracy": round(phone_accuracy, 3),
                "date_format_accuracy": round(date_accuracy, 3),
                "phone_rows_correct": phone_ok,
                "date_rows_correct": date_ok,
            },
        }

    def grade(self, rows: List[Dict[str, Any]]) -> Tuple[float, bool]:
        evaluation = self.evaluate(rows)
        return _clamp(evaluation["raw_score"]), bool(evaluation["task_complete"])


class HardGrader:
    SALARY_THRESHOLD = 300000

    def _is_outlier(self, salary: Any) -> bool:
        try:
            return float(salary) > self.SALARY_THRESHOLD
        except (TypeError, ValueError):
            return False

    def evaluate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        seen = set()
        duplicate_rows = 0
        for r in rows:
            key = (r.get("name"), r.get("dept"), r.get("salary"))
            if key in seen:
                duplicate_rows += 1
            else:
                seen.add(key)

        outlier_rows = sum(1 for r in rows if self._is_outlier(r.get("salary")))
        no_dup = duplicate_rows == 0
        no_outlier = outlier_rows == 0
        raw = (0.5 * int(no_dup)) + (0.5 * int(no_outlier))
        done = no_dup and no_outlier
        return {
            "raw_score": raw,
            "task_complete": done,
            "details": {
                "duplicate_rows_remaining": duplicate_rows,
                "outlier_rows_remaining": outlier_rows,
            },
        }

    def grade(self, rows: List[Dict[str, Any]]) -> Tuple[float, bool]:
        evaluation = self.evaluate(rows)
        return _clamp(evaluation["raw_score"]), bool(evaluation["task_complete"])


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
        self._step = 0
        self._score = _clamp(0.0)
        self._done = False
        self._initial_missing_values = 0
        self._initial_row_count = 0
        self._initial_raw_score = 0.0

    def reset(self) -> DataCleaningObservation:
        self._rows = copy.deepcopy(DATASET_FACTORIES[self._task_name]())
        self._step = 0
        self._done = False
        evaluation = self._grader.evaluate(self._rows)
        self._initial_raw_score = float(evaluation["raw_score"])
        self._initial_missing_values = self._count_missing_values(self._rows)
        self._initial_row_count = len(self._rows)
        score, _ = self._grader.grade(self._rows)
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

        graded_evaluation = self._grader.evaluate(self._rows)
        graded_score = _clamp(float(graded_evaluation["raw_score"]))
        graded_done = bool(graded_evaluation["task_complete"])
        self._score = graded_score

        if action.action_type == "done":
            if not graded_done:
                reward -= PREMATURE_DONE_PENALTY
                result_msg = (
                    f"Done signalled before task completion. "
                    f"Final score: {graded_score:.2f}"
                )
            else:
                result_msg = f"Done signalled. Task complete at {graded_score:.2f}"
            self._done = True

        if self._step >= self._task["max_steps"]:
            self._done = True
            result_msg += " (max steps reached)"

        info = {
            "step": self._step,
            "graded_score": graded_score,
            "action_type": action.action_type,
            "task_complete": graded_done,
            "evaluation": self._evaluation_metrics(graded_evaluation),
        }

        return StepResult(
            observation=self._make_obs(result_msg),
            reward=reward,
            done=self._done,
            info=info,
        )

    @property
    def state(self) -> DataCleaningState:
        return DataCleaningState(
            task_name=self._task_name,
            step=self._step,
            score=self._score,
            done=self._done,
        )

    @property
    def evaluation_metrics(self) -> Dict[str, Any]:
        return self._evaluation_metrics()

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
            return "Done signal received.", 0.0
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

    def _count_missing_values(self, rows: List[Dict[str, Any]]) -> int:
        return sum(1 for row in rows for value in row.values() if value is None)

    def _evaluation_metrics(self, graded_evaluation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if graded_evaluation is None:
            graded_evaluation = self._grader.evaluate(self._rows)

        raw_score = float(graded_evaluation["raw_score"])
        missing_current = self._count_missing_values(self._rows)
        missing_initial = self._initial_missing_values
        if missing_initial > 0:
            missing_reduction = (missing_initial - missing_current) / missing_initial
        else:
            missing_reduction = 0.0

        quality_gain = raw_score - self._initial_raw_score
        metrics = {
            "data_quality_score": round(max(0.0, min(1.0, raw_score)) * 100.0, 2),
            "quality_gain": round(quality_gain * 100.0, 2),
            "missing_values_initial": missing_initial,
            "missing_values_current": missing_current,
            "missing_value_reduction": round(max(0.0, min(1.0, missing_reduction)), 3),
            "rows_initial": self._initial_row_count,
            "rows_current": len(self._rows),
            "rows_removed": max(0, self._initial_row_count - len(self._rows)),
        }

        details = graded_evaluation.get("details", {})
        if details:
            metrics["task_details"] = details
        return metrics

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
        score, _ = self._grader.grade(self._rows) if self._rows else (_clamp(0.0), False)
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
