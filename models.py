"""
Clinical Trial Optimization — OpenEnv Models
Typed Pydantic models that extend the official openenv-core base classes.

Covers 5 distinct clinical trial management tasks with rich observation
and action spaces modeling real pharmaceutical R&D decision-making.
"""
from typing import Dict, List, Optional

from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class TrialAction(Action):
    """What the AI agent can do each week in the clinical trial."""

    decision: str = Field(
        ...,
        description=(
            "One of: increase_dose, decrease_dose, keep_dose, "
            "enroll_more_patients, stop_trial, add_treatment_arm, "
            "drop_treatment_arm, request_interim_analysis, "
            "pause_enrollment, adjust_monitoring"
        ),
    )


class TrialObservation(Observation):
    """Rich observation space combining clinical, statistical, and regulatory signals."""

    # ── Core trial status ──
    week: int = Field(..., description="Current week of the trial (1-52)")
    current_dose_mg: int = Field(..., description="Current dose in mg")
    patients_enrolled: int = Field(..., description="Total patients enrolled")
    avg_effectiveness: float = Field(
        ..., description="Average primary endpoint improvement (0.0-1.0)"
    )
    side_effect_rate: float = Field(
        ..., description="Fraction of patients with adverse events (0.0-1.0)"
    )
    budget_remaining: float = Field(..., description="Remaining budget in USD")
    message: str = Field(default="", description="Human-readable status message")

    # ── Statistical signals ──
    confidence_interval_low: float = Field(
        default=0.0, description="Lower bound of 95% CI for effectiveness"
    )
    confidence_interval_high: float = Field(
        default=0.0, description="Upper bound of 95% CI for effectiveness"
    )
    statistical_power: float = Field(
        default=0.0, description="Current statistical power (0.0-1.0)"
    )

    # ── Patient dynamics ──
    dropout_rate: float = Field(
        default=0.0, description="Patient attrition rate (0.0-1.0)"
    )
    active_treatment_arms: int = Field(
        default=1, description="Number of active treatment arms"
    )
    patients_per_arm: List[int] = Field(
        default_factory=lambda: [0], description="Patients in each arm"
    )

    # ── Safety signals ──
    serious_adverse_events: int = Field(
        default=0, description="Cumulative serious adverse event count"
    )
    organ_toxicity: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-organ toxicity levels (liver, kidney, cardiac)",
    )

    # ── Regulatory & clinical signals ──
    interim_analysis_due: bool = Field(
        default=False, description="Is an interim analysis scheduled?"
    )
    data_maturity: float = Field(
        default=0.0, description="Fraction of planned follow-up completed (0.0-1.0)"
    )
    futility_probability: float = Field(
        default=0.0,
        description="Bayesian predictive probability of trial failure (0.0-1.0)",
    )

    # Note: `done` and `reward` are inherited from Observation base class


class TrialState(State):
    """Full metadata about the current episode — extends OpenEnv State."""

    # episode_id and step_count are inherited from State base class
    task_name: str = Field(
        default="", description="Name of current task being run"
    )
    difficulty: str = Field(
        default="easy", description="Task difficulty: easy/medium/hard"
    )
    true_optimal_dose: int = Field(
        default=0, description="Hidden optimal dose (not shown to agent)"
    )
    safe_dose_limit: int = Field(
        default=0, description="Hidden safety threshold dose"
    )

    # ── Trajectory tracking for grader ──
    doses_visited: List[int] = Field(
        default_factory=list, description="All doses tried during episode"
    )
    max_side_effects_seen: float = Field(
        default=0.0, description="Worst side effect rate during episode"
    )
    max_effectiveness_seen: float = Field(
        default=0.0, description="Best effectiveness achieved"
    )
    interim_analyses_requested: int = Field(
        default=0, description="Number of interim analyses run"
    )
    arms_added: int = Field(default=0, description="Treatment arms added")
    arms_dropped: int = Field(default=0, description="Treatment arms dropped")
    enrollment_paused: bool = Field(
        default=False, description="Whether enrollment was paused for safety"
    )
    safety_responses: int = Field(
        default=0, description="Times agent responded to safety signals"
    )
    sae_count: int = Field(
        default=0, description="Total serious adverse events during episode"
    )