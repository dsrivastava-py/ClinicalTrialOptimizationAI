"""
Clinical Trial Optimization — FastAPI Server
Uses openenv-core's create_fastapi_app() to build an OpenEnv-compliant server.
Custom endpoints (/tasks, /grade, /reset/<task>) are layered on top.
"""
import sys
import os

# Make root importable regardless of how uvicorn is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app
from models import TrialAction, TrialObservation
from server.environment import ClinicalTrialEnvironment, TASK_CONFIG
from server.grader import grade_by_task

# ── OPENENV-COMPLIANT APP ────────────────────────────────────────────────────
# create_fastapi_app sets up /reset, /step, /state, /health, /schema, /ws
app: FastAPI = create_fastapi_app(ClinicalTrialEnvironment, TrialAction, TrialObservation)

# ── SHARED ENV for stateful grade/task endpoints ─────────────────────────────
_shared_env = ClinicalTrialEnvironment()


# ── TASK-SPECIFIC RESET ENDPOINTS ────────────────────────────────────────────

def _reset_task(task_name: str):
    """Helper: reset shared env for a specific task."""
    config = TASK_CONFIG.get(task_name, {})
    _shared_env.difficulty = config.get("difficulty", "easy")
    obs = _shared_env.reset(task_name=task_name)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


@app.post("/reset/dose_escalation")
def reset_dose_escalation():
    """Start a DOSE ESCALATION episode (easy — find optimal dose via 3+3 design)."""
    return _reset_task("dose_escalation")


@app.post("/reset/adaptive_enrollment")
def reset_adaptive_enrollment():
    """Start an ADAPTIVE ENROLLMENT episode (medium — manage treatment arms)."""
    return _reset_task("adaptive_enrollment")


@app.post("/reset/interim_analysis")
def reset_interim_analysis():
    """Start an INTERIM ANALYSIS episode (medium — continue/stop decisions)."""
    return _reset_task("interim_analysis")


@app.post("/reset/safety_monitoring")
def reset_safety_monitoring():
    """Start a SAFETY MONITORING episode (hard — detect organ toxicity signals)."""
    return _reset_task("safety_monitoring")


@app.post("/reset/multi_endpoint")
def reset_multi_endpoint():
    """Start a MULTI-ENDPOINT episode (hard — optimize primary + secondary endpoints)."""
    return _reset_task("multi_endpoint")


# Keep legacy /reset/<difficulty> endpoints for backward compatibility
@app.post("/reset/easy")
def reset_easy():
    """Alias: Start dose_escalation (easy)."""
    return _reset_task("dose_escalation")


@app.post("/reset/medium")
def reset_medium():
    """Alias: Start interim_analysis (medium)."""
    return _reset_task("interim_analysis")


@app.post("/reset/hard")
def reset_hard():
    """Alias: Start safety_monitoring (hard)."""
    return _reset_task("safety_monitoring")


# ── TASKS LISTING ────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """List all 5 available tasks with descriptions, difficulties, and endpoints."""
    tasks = []
    for name, config in TASK_CONFIG.items():
        tasks.append({
            "name": name,
            "difficulty": config["difficulty"],
            "description": config["description"],
            "max_steps": config["max_steps"],
            "success_threshold": 0.5,
            "reset_endpoint": f"/reset/{name}",
        })
    return JSONResponse(content={"tasks": tasks})


# ── GRADING ENDPOINT ─────────────────────────────────────────────────────────

@app.post("/grade")
def grade():
    """
    Grade the current shared episode.
    Returns score 0.0-1.0 with detailed per-criterion reasons.
    """
    task_name = _shared_env.state.task_name if _shared_env.state else ""
    score, reasons = grade_by_task(_shared_env, task_name)
    return JSONResponse(content={
        "score": score,
        "task": task_name,
        "reasons": reasons,
    })


# ── ROOT INFO ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint — environment info and available endpoints."""
    return JSONResponse(content={
        "name": "clinical_trial_env",
        "version": "0.2.0",
        "description": (
            "Clinical trial optimization RL environment with 5 distinct tasks "
            "(dose escalation, adaptive enrollment, interim analysis, "
            "safety monitoring, multi-endpoint optimization). OpenEnv compliant."
        ),
        "framework": "openenv-core",
        "tasks": list(TASK_CONFIG.keys()),
        "endpoints": [
            "/reset", "/step", "/state", "/health",
            "/schema", "/ws",
            "/reset/dose_escalation", "/reset/adaptive_enrollment",
            "/reset/interim_analysis", "/reset/safety_monitoring",
            "/reset/multi_endpoint",
            "/tasks", "/grade",
        ],
    })


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()