---
title: Clinical Trial Optimization AI
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Clinical Trial Optimization — OpenEnv RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An RL environment where an AI agent acts as a clinical trial manager,
making sequential decisions across **5 distinct pharmaceutical R&D tasks**
— from dose escalation to multi-endpoint optimization. Designed to train
and evaluate AI agents on real-world clinical decision-making.

Built with [`openenv-core`](https://pypi.org/project/openenv-core/) — implements
the full OpenEnv spec including `step()`, `reset()`, `state()`, and `openenv.yaml`.

---

## Real-World Problem

Clinical trials cost **\$1–3 billion** and take **10–15 years** on average.
**90% of drugs fail** somewhere in this process — often due to suboptimal
sequential decision-making about dosage, patient enrollment, safety
monitoring, and when to stop. This environment trains AI agents to make
these decisions better, filling a genuine gap in RL research.

Unlike simple dose-finding simulators, this environment models the
**full complexity of Phase 2 trial management**: adaptive designs,
interim analyses, multi-organ safety monitoring, multi-endpoint
optimization, and Bayesian statistical reasoning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Agent (LLM)                           │
│  Observes: dose, effectiveness, safety, power, toxicity     │
│  Actions:  10 clinical trial management decisions           │
└────────────────────────┬────────────────────────────────────┘
                         │ step(action)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Clinical Trial Environment                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Dose-Response │  │ Safety Model │  │ Statistical Engine│ │
│  │  (Bell curve) │  │  (Sigmoid)   │  │ (Power, CI, etc.) │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │  Patient      │  │ Organ        │  │ Budget & Time     │ │
│  │  Dynamics     │  │ Toxicity     │  │ Management        │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ grade()
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Per-Task Grader (0.0 – 1.0)                    │
│  6+ criteria per task: safety, effectiveness, exploration,  │
│  dose accuracy, efficiency, budget, arm management, etc.    │
└─────────────────────────────────────────────────────────────┘
```

---

## 5 Tasks

| # | Task | Difficulty | Key Challenge | Max Steps |
|---|------|------------|---------------|-----------|
| 1 | `dose_escalation` | Easy | Classic 3+3 dose-finding with DLT monitoring | 20 |
| 2 | `adaptive_enrollment` | Medium | Bayesian adaptive randomization across treatment arms | 25 |
| 3 | `interim_analysis` | Medium | Futility/efficacy stopping decisions at interim looks | 30 |
| 4 | `safety_monitoring` | Hard | DSMB-style multi-organ safety signal detection | 25 |
| 5 | `multi_endpoint` | Hard | Primary + secondary endpoint trade-off optimization | 30 |

Each task has a **unique grading rubric** with 6+ independently scored criteria
that evaluate trajectory quality (exploration, safety responses, adaptive
management), not just final outcomes.

---

## Action Space (10 Actions)

| Action | Description | Cost |
|--------|-------------|------|
| `increase_dose` | Escalate dose by 20mg | \$150,000 |
| `decrease_dose` | De-escalate dose by 15mg | \$100,000 |
| `keep_dose` | Maintain dose, collect data | \$80,000 |
| `enroll_more_patients` | Add 40 patients | \$200,000 |
| `stop_trial` | Conclude trial and submit results | — |
| `add_treatment_arm` | Add new dose arm (adaptive) | \$250,000 |
| `drop_treatment_arm` | Drop worst-performing arm | — |
| `request_interim_analysis` | Formal interim review | \$120,000 |
| `pause_enrollment` | Pause/resume for safety review | \$50,000 |
| `adjust_monitoring` | Enhance safety monitoring | \$60,000 |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `week` | int | Current trial week (1–52) |
| `current_dose_mg` | int | Current dose in mg |
| `patients_enrolled` | int | Total patients |
| `avg_effectiveness` | float | Primary endpoint improvement (0–1) |
| `side_effect_rate` | float | Adverse event fraction (0–1) |
| `budget_remaining` | float | Remaining budget in USD |
| `confidence_interval_low` | float | Lower 95% CI for effectiveness |
| `confidence_interval_high` | float | Upper 95% CI for effectiveness |
| `statistical_power` | float | Current statistical power (0–1) |
| `dropout_rate` | float | Patient attrition rate |
| `active_treatment_arms` | int | Number of active arms |
| `serious_adverse_events` | int | Cumulative SAE count |
| `organ_toxicity` | dict | Per-organ toxicity (liver, kidney, cardiac) |
| `interim_analysis_due` | bool | Interim analysis scheduled? |
| `data_maturity` | float | Follow-up completion fraction |
| `futility_probability` | float | Bayesian failure probability |
| `done` | bool | Episode ended |
| `reward` | float | Step reward (-1 to 1) |

---

## Reward Function

Shaped rewards normalised to **[-1.0, +1.0]** per step:

| Signal | Value | Condition |
|--------|-------|-----------|
| Strong effectiveness found | +0.25 | Effectiveness > 40% |
| Moderate improvement | +0.10 | Effectiveness > 25% |
| Successful trial stop | +0.50 | Effective + safe dose confirmed |
| Safety response | +0.10 to +0.20 | Appropriate pause/decrease on signal |
| Adaptive arm management | +0.05 to +0.10 | Smart arm additions/drops |
| Interim analysis | +0.05 to +0.15 | Timely interim review |
| Safety violation | -0.30 | Side effects > 30% |
| Emergency stop | -0.40 | Side effects > 50% |
| Budget exhaustion | -0.20 | Budget runs out |

## Grading (0.0 – 1.0)

Each task has 6+ independent criteria. Example for `dose_escalation`:

| Criterion | Weight | Condition |
|-----------|--------|-----------|
| Safety maintained | 0.20 | max side_effect_rate ≤ 30% |
| Effectiveness | 0.25 | effectiveness ≥ 30% |
| Dose exploration | 0.15 | ≥3 different doses tested |
| Dose accuracy | 0.20 | within 10mg of optimal |
| Efficiency | 0.10 | found dose within 15 weeks |
| Budget managed | 0.10 | budget not exhausted |

---

## Setup

```bash
# Install dependencies (includes openenv-core)
pip install -r requirements.txt

# Run locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Verify environment
python verify.py
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

# Run all 5 tasks
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (generic) |
| `/reset/dose_escalation` | POST | Start dose escalation task |
| `/reset/adaptive_enrollment` | POST | Start adaptive enrollment task |
| `/reset/interim_analysis` | POST | Start interim analysis task |
| `/reset/safety_monitoring` | POST | Start safety monitoring task |
| `/reset/multi_endpoint` | POST | Start multi-endpoint task |
| `/step` | POST | Take one action |
| `/state` | GET | Current episode metadata |
| `/ws` | WebSocket | Full WebSocket interface |
| `/tasks` | GET | List all 5 tasks |
| `/grade` | POST | Grade current episode (0.0–1.0) |
| `/health` | GET | Health check |
| `/schema` | GET | Action/Observation JSON schemas |

---

## Client Usage

```python
from client import ClinicalTrialEnvClient, TrialAction

with ClinicalTrialEnvClient("http://localhost:7860") as client:
    # List tasks
    tasks = client.tasks()

    # Run dose escalation
    obs = client.reset("dose_escalation")
    obs = client.step(TrialAction(decision="increase_dose"))
    obs = client.step(TrialAction(decision="increase_dose"))
    obs = client.step(TrialAction(decision="stop_trial"))

    # Grade the episode
    result = client.grade()
    print(f"Score: {result['score']}")
```

---

## Project Structure

```
clinical_trial_env/
├── .dockerignore           # Docker build exclusions
├── __init__.py             # Module exports (empty)
├── client.py               # HTTP client for programmatic access
├── models.py               # TrialAction, TrialObservation, TrialState
├── inference.py            # Baseline inference script (OpenEnv spec)
├── openenv.yaml            # OpenEnv manifest (5 tasks, 10 actions)
├── pyproject.toml          # Project metadata
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image definition
├── README.md               # This file
├── verify.py               # Automated verification tests
└── server/
    ├── __init__.py         # Server module
    ├── app.py              # FastAPI + openenv create_fastapi_app
    ├── environment.py      # Core simulation logic (5 task types)
    └── grader.py           # Per-task deterministic graders (6+ criteria)
```

---

## OpenEnv Compliance

- ✅ `openenv-core` base classes (`Action`, `Observation`, `State`, `Environment`)
- ✅ `step(action)` → returns observation, reward, done
- ✅ `reset()` → returns initial observation
- ✅ `state` property → returns episode metadata
- ✅ `openenv.yaml` with `spec_version: 1`, `tags: [openenv]`
- ✅ 5 tasks with distinct grading rubrics (6+ criteria each)
- ✅ Meaningful trajectory-aware reward shaping
- ✅ `create_fastapi_app()` from openenv-core
- ✅ WebSocket endpoint (`/ws`)
- ✅ Working `Dockerfile` (2 vCPU / 8GB RAM compliant)
- ✅ Baseline `inference.py` with `[START]`/`[STEP]`/`[END]` log format
- ✅ `client.py` for programmatic access
- ✅ `.dockerignore` for lean builds