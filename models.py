"""
Clinical Trial Optimization — OpenEnv Models
Typed Pydantic models that extend the official openenv-core base classes.
"""
from typing import Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class TrialAction(Action):
    """What the AI can do each week in the clinical trial."""

    decision: str = Field(
        ...,
        description=(
            "One of: increase_dose, decrease_dose, keep_dose, "
            "enroll_more_patients, stop_trial"
        ),
    )


class TrialObservation(Observation):
    """What the AI can see each week — current trial status."""

    week: int = Field(..., description="Current week of the trial (1-52)")
    current_dose_mg: int = Field(..., description="Current dose being tested in mg")
    patients_enrolled: int = Field(..., description="Total patients in trial so far")
    avg_effectiveness: float = Field(
        ..., description="Average blood sugar reduction (0.0-1.0)"
    )
    side_effect_rate: float = Field(
        ..., description="Fraction of patients with side effects (0.0-1.0)"
    )
    budget_remaining: float = Field(..., description="Remaining budget in USD")
    message: str = Field(default="", description="Human readable description of what happened")
    # Note: `done` and `reward` are inherited from Observation base class


class TrialState(State):
    """Metadata about the current episode — extends OpenEnv State."""

    # episode_id and step_count are inherited from State base class
    task_name: str = Field(default="", description="Name of current task being run")
    difficulty: str = Field(default="easy", description="Task difficulty: easy/medium/hard")
    true_optimal_dose: int = Field(default=0, description="Hidden optimal dose (not shown to agent)")
    safe_dose_limit: int = Field(default=0, description="Hidden safety threshold dose")