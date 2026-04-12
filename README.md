---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Data Cleaning OpenEnv

A real-world [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an AI agent cleans dirty datasets through sequential actions.

## Description

An agent is given a dirty CSV-style dataset and must clean it by calling actions step-by-step. The environment rewards correct fixes and penalises wrong or redundant actions. Three tasks of increasing difficulty test different data-cleaning skills.

## Action Space

| action_type | Required fields | Effect |
|-------------|----------------|--------|
| `fill_null` | `column`, `value` | Fill all null entries in `column` with `value` |
| `replace` | `column`, `old`, `new` | Replace `old` with `new` in `column` |
| `drop_row` | `index` | Drop the row whose `id` == `index` |
| `done` | — | Signal episode complete; score is finalised |

## Observation Space

```json
{
  "task_name": "easy",
  "task_description": "...",
  "rows": [{"id": 1, "name": "Alice", "age": 30}, ...],
  "column_stats": [{"name": "age", "dtype": "int", "null_count": 3, ...}],
  "step": 1,
  "max_steps": 10,
  "last_action_result": "Filled 3 null(s) in 'age' with 0",
  "score": 1.0,
  "done": false
}
```

## Tasks

### Easy — Fill Missing Values
Fill all `null` values in the `age` column with `0`.  
Score = fraction of rows where `age` is non-null.  
Max steps: 10 | Baseline score: 0.0

### Medium — Normalize Formats
Normalize `phone` to `(XXX) XXX-XXXX` and `date` to `YYYY-MM-DD`.  
Score = (correct phones + correct dates) / (2 × rows).  
Max steps: 20 | Baseline score: 0.5

### Hard — Remove Duplicates & Outliers
Remove exact duplicate rows and rows where `salary > 300,000`.  
Score = 0.5 for no duplicates + 0.5 for no outliers.  
Max steps: 30 | Baseline score: 0.0

## Reward Design

| Action outcome | Reward |
|---------------|--------|
| `fill_null` fills N nulls | `+0.1 × N` |
| `replace` changes N values | `+0.15 × N` |
| `drop_row` removes a row | `+0.20` |
| Nothing changed | `-0.02` |
| Invalid action | `-0.05` |
| `done` | `0.0` (score finalised) |

## Setup

### Local (uvicorn)
```bash
pip install -r requirements.txt
TASK_NAME=easy uvicorn server.app:main --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
docker build -t data-cleaning-env .
docker run -d -p 8000:8000 -e TASK_NAME=medium data-cleaning-env
```

### Hugging Face Spaces
```bash
openenv push --repo-id <username>/data-cleaning-env
```

## Quick Test
```bash
# Reset
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'

# Fill nulls (easy task)
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "fill_null", "column": "age", "value": 0}'

# Signal done
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "done"}'

# Check health
curl http://localhost:8000/health
```

## Run Inference
```bash
export HF_TOKEN=hf_...
export ENV_BASE_URL=https://<your-space>.hf.space
export TASK_NAME=easy
python inference.py
```

## Baseline Scores

| Task | Random agent | Rule-based | Notes |
|------|-------------|------------|-------|
| easy | 0.0 | 1.0 | One fill_null action solves it |
| medium | 0.5 | 1.0 | Requires format-aware replace actions |
| hard | 0.0 | 1.0 | Must identify duplicates + outliers |

## Project Structure
```
data_cleaning_env/
├── inference.py          ← MANDATORY root inference script
├── openenv.yaml          ← OpenEnv manifest
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── README.md
├── models.py             ← Typed Action/Observation/State
└── server/
    ├── app.py            ← FastAPI server
    └── environment.py    ← Task logic + graders
```

# 🚀 Enhanced Hackathon Presentation

## Problem Statement (Real-World Impact)
Dirty tabular data slows analytics, breaks dashboards, and causes bad decisions in operations, finance, and ML pipelines. Teams lose hours on repetitive cleaning steps that are ideal for autonomous agents.

## Solution (What This Environment Proves)
This OpenEnv submission turns data cleaning into a sequential decision task where an AI agent must:
1. Inspect dataset state
2. Choose one safe cleaning action
3. Observe reward + quality metrics
4. Iterate until quality is restored

It demonstrates measurable improvement, not just action execution.

## AI Agent Behavior (Judge-Friendly)
- `easy`: learns to fill nulls in the right column only
- `medium`: learns format normalization for phone/date fields
- `hard`: learns structural cleanup by removing duplicates and outliers
- Agent is discouraged from premature termination (`done` before completion is penalized)

## Architecture (Simple Flow)
`Agent -> /reset -> observation(rows + stats + score) -> /step(action) -> reward + updated observation + evaluation metrics -> repeat -> done`

## Evaluation Metrics (Visible in API `info.evaluation`)
- `data_quality_score` (0-100)
- `quality_gain` (improvement from initial state)
- `missing_value_reduction` (0-1)
- `rows_removed`
- Task-specific diagnostics (for example: `phone_format_accuracy`, `duplicate_rows_remaining`)

## Example Results (Before vs After)
| Task | Before | After | Outcome |
|------|--------|-------|---------|
| easy | 3 missing `age` values, quality `40.0` | 0 missing, quality `100.0` | Fully cleaned |
| medium | mixed phone/date formats, quality `50.0` | normalized formats, quality `100.0` | Fully standardized |
| hard | duplicates + salary outliers, quality `0.0` | no duplicates/outliers, quality `100.0` | Fully sanitized |

## Demo in <10 Seconds
1. Open Space URL (or local server root) and verify health: `GET /health`
2. Start episode: `POST /reset {"task_name":"easy"}`
3. Clean once: `POST /step {"action_type":"fill_null","column":"age","value":0}`
4. Finalize: `POST /step {"action_type":"done"}`
5. Show judge: `observation.score` + `info.evaluation`

## Hugging Face Demo UX Tips
- Use `easy` task first for instant win in one action
- Highlight `info.evaluation.missing_value_reduction` jumping to `1.0`
- Show root endpoint (`/`) quick-start payload for immediate API guidance
- For reliability demo, run `python test_local.py` before `openenv push`

## Sample Input / Output Snapshot
Input action:
```json
{"action_type":"fill_null","column":"age","value":0}
```
Output highlights:
```json
{
  "reward": 0.3,
  "observation": {"score": 0.99},
  "info": {
    "evaluation": {
      "data_quality_score": 100.0,
      "missing_value_reduction": 1.0
    }
  }
}
```

