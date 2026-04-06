# Clinical Trial Optimization — OpenEnv RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)

An RL environment where an AI agent acts as a clinical trial manager,
making sequential decisions to find the optimal drug dose safely,
efficiently, and within budget.

Built with [`openenv-core`](https://pypi.org/project/openenv-core/) — implements
the full OpenEnv spec including `step()`, `reset()`, `state()`, and `openenv.yaml`.

---

## Real-World Problem

Clinical trials cost \$1–3 billion and take 10–15 years on average.
90% of drugs fail somewhere in this process — often due to suboptimal
sequential decision-making about dosage, patient enrollment, and when
to stop. This environment trains AI agents to make these decisions
better, filling a genuine gap in RL research.

## Environment Description

The agent manages a simulated Phase 2 drug trial for a Type 2 Diabetes
medication. Each episode, a drug with a **hidden** optimal dose is generated.
The agent must discover this dose through experimentation while:

- Staying within safety thresholds (side effects < 30%)
- Finding an effective dose (effectiveness > 35%)
- Managing a \$5M budget
- Concluding the trial efficiently (within 52 weeks)

The environment randomises the true optimal dose and safety margin each
episode, making it a genuine learning problem — not a lookup table.

---

## Action Space

| Action | Description | Cost |
|--------|-------------|------|
| `increase_dose` | Raise current dose by 20mg | \$150,000 |
| `decrease_dose` | Lower current dose by 15mg | \$100,000 |
| `keep_dose` | Maintain dose, collect more data | \$80,000 |
| `enroll_more_patients` | Add 40 more patients at current dose | \$200,000 |
| `stop_trial` | Conclude the trial and submit results | — |

## Observation Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `week` | int | 1–52 | Current trial week |
| `current_dose_mg` | int | 5–150 | Current dose being tested |
| `patients_enrolled` | int | — | Total patients in trial |
| `avg_effectiveness` | float | 0–1 | Blood sugar reduction rate |
| `side_effect_rate` | float | 0–1 | Fraction with side effects |
| `budget_remaining` | float | — | Remaining USD budget |
| `done` | bool | — | Episode ended |
| `reward` | float | -1–1 | Reward for last action |
| `message` | str | — | Human-readable outcome |

---

## Tasks

| Task | Difficulty | Optimal Dose Range | Safety Margin | Max Steps |
|------|------------|-------------------|---------------|-----------|
| `dose_finding_easy` | Easy | 30–70mg | 20–35mg | 20 |
| `dose_finding_medium` | Medium | 50–80mg | 13–18mg | 20 |
| `dose_finding_hard` | Hard | 60–100mg | 8–12mg | 20 |

Each task has a **programmatic grader** that returns a score from 0.0–1.0.
Tasks progress from easy (wide safety window) to hard (very narrow margin requiring
precise, conservative dose titration).

---

## Reward Function

Shaped rewards normalised to **[-1.0, +1.0]** per step:

| Signal | Value | Condition |
|--------|-------|-----------|
| Found strong effectiveness | +0.25 | Effectiveness > 40% |
| Moving toward effective dose | +0.10 | Effectiveness > 25% |
| Good safety decision | +0.15 | Side effects < 15% and effectiveness > 30% |
| Perfect trial stop | +0.50 | Stopped with effectiveness > 35% and side effects < 20% |
| Safety violation | -0.30 | Side effects > 30% |
| Dangerous dose | -0.40 | Side effects > 50% emergency |
| Budget exhaustion | -0.20 | Budget runs out |

## Grader Scoring (0.0 – 1.0)

| Criterion | Score | Condition |
|-----------|-------|-----------|
| Safety maintained | +0.25 | side_effect_rate ≤ 30% |
| Strong effectiveness | +0.30 | effectiveness ≥ 40% |
| Moderate effectiveness | +0.15 | effectiveness ≥ 25% |
| Efficiency bonus | +0.20 | Found dose within ≤20 weeks |
| Budget not exhausted | +0.10 | Budget remaining > 0 |
| Budget efficiency | +0.15 | Used <50% of budget when effective |

---

## Setup

```bash
# Install dependencies (includes openenv-core)
pip install -r requirements.txt

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t clinical-trial-env .
docker run -p 7860:7860 clinical-trial-env
```

## Baseline Inference

```bash
# Set your credentials
export HF_TOKEN=hf_your_key_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

# Run all 3 tasks
python inference.py
```

## Baseline Scores (llama-3.1-8b-instant via Groq)

*Run with `HF_TOKEN=<groq_key>`, `API_BASE_URL=https://api.groq.com/openai/v1`, `MODEL_NAME=llama-3.1-8b-instant`*

| Task | Score | Notes |
|------|-------|-------|
| dose_finding_easy | 0.650 | Safety maintained, effectiveness ≥40% |
| dose_finding_medium | 0.350 | Safety maintained, dose not explored sufficiently |
| dose_finding_hard | 0.332 | Safety maintained, dose not explored sufficiently |
| **Average** | **0.444** | |

> These are conservative baseline scores. A smarter agent that properly explores the dose space
> should achieve 0.65+ on medium and 0.50+ on hard tasks.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (generic) |
| `/reset/easy` | POST | Start easy difficulty episode |
| `/reset/medium` | POST | Start medium difficulty episode |
| `/reset/hard` | POST | Start hard difficulty episode |
| `/step` | POST | Take one action |
| `/state` | GET | Current episode metadata |
| `/ws` | WebSocket | Full WebSocket interface |
| `/tasks` | GET | List all available tasks |
| `/grade` | POST | Grade current episode (0.0–1.0) |
| `/health` | GET | Health check |
| `/schema` | GET | Action/Observation JSON schemas |

---

## OpenEnv Compliance

This environment fully implements the OpenEnv spec:

- ✅ `openenv-core` base classes (`Action`, `Observation`, `State`, `Environment`)
- ✅ `step(action)` → returns observation, reward, done
- ✅ `reset()` → returns initial observation
- ✅ `state` property → returns episode metadata
- ✅ `openenv.yaml` with `spec_version: 1`, `tags: [openenv]`
- ✅ 3 tasks (easy → medium → hard) with agent graders (0.0–1.0)
- ✅ Meaningful shaped reward function
- ✅ `create_fastapi_app()` from openenv-core
- ✅ WebSocket endpoint (`/ws`)
- ✅ Working `Dockerfile`
- ✅ Baseline `inference.py` with `[START]`/`[STEP]`/`[END]` log format