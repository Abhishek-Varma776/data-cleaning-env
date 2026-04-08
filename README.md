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

