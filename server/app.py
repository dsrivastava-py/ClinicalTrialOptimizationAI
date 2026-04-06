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
from server.environment import ClinicalTrialEnvironment
from server.grader import grade_by_task

# ── OPENENV-COMPLIANT APP ────────────────────────────────────────────────────
# create_fastapi_app sets up /reset, /step, /state, /health, /schema, /ws
# It accepts a callable factory so each HTTP request gets a fresh env instance
app: FastAPI = create_fastapi_app(ClinicalTrialEnvironment, TrialAction, TrialObservation)

# ── SINGLETON ENV for stateful grade/task endpoints ──────────────────────────
# The OpenEnv HTTP endpoints each spin up a new env per call (stateless).
# We maintain a shared env for /grade and task-specific /reset/<task> routes.
_shared_env = ClinicalTrialEnvironment()


# ── TASK-SPECIFIC RESET ENDPOINTS ────────────────────────────────────────────

@app.post("/reset/easy")
def reset_easy():
    """Start a new EASY episode (30-70mg optimal dose, wide safety window)."""
    _shared_env.difficulty = "easy"
    obs = _shared_env.reset(task_name="dose_finding_easy")
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


@app.post("/reset/medium")
def reset_medium():
    """Start a new MEDIUM episode (50-80mg optimal dose, narrower safety margin)."""
    _shared_env.difficulty = "medium"
    obs = _shared_env.reset(task_name="dose_finding_medium")
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


@app.post("/reset/hard")
def reset_hard():
    """Start a new HARD episode (60-100mg optimal dose, very narrow safety window)."""
    _shared_env.difficulty = "hard"
    obs = _shared_env.reset(task_name="dose_finding_hard")
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })


# ── TASKS LISTING ────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """List all available tasks with their descriptions and difficulty."""
    return JSONResponse(content={
        "tasks": [
            {
                "name": "dose_finding_easy",
                "difficulty": "easy",
                "description": (
                    "Find optimal dose (30-70mg range) with wide safety window. "
                    "Side effect threshold is generous. Success requires "
                    "effectiveness > 35% without exceeding 30% side effect rate."
                ),
                "max_steps": 20,
                "success_threshold": 0.5,
                "reset_endpoint": "/reset/easy",
            },
            {
                "name": "dose_finding_medium",
                "difficulty": "medium",
                "description": (
                    "Find optimal dose (50-80mg range) with narrower safety margin. "
                    "Requires more precise titration and careful balance of "
                    "effectiveness vs. safety."
                ),
                "max_steps": 20,
                "success_threshold": 0.5,
                "reset_endpoint": "/reset/medium",
            },
            {
                "name": "dose_finding_hard",
                "difficulty": "hard",
                "description": (
                    "Find optimal dose (60-100mg range) with very narrow safety window "
                    "(only 8-12mg above optimal before side effects spike). "
                    "Requires precise, conservative exploration."
                ),
                "max_steps": 20,
                "success_threshold": 0.5,
                "reset_endpoint": "/reset/hard",
            },
        ]
    })


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
    """Root endpoint — returns environment info and available endpoints."""
    return JSONResponse(content={
        "name": "clinical_trial_env",
        "version": "0.1.0",
        "description": "Clinical drug trial optimization RL environment (OpenEnv compliant)",
        "framework": "openenv-core",
        "endpoints": [
            "/reset", "/step", "/state", "/health",
            "/schema", "/ws",
            "/reset/easy", "/reset/medium", "/reset/hard",
            "/tasks", "/grade",
        ],
    })