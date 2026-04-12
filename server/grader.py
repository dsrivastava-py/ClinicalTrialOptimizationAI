"""
Clinical Trial Optimization — Per-Task Grader
Each task has its own deterministic rubric with 6+ independently scored criteria.
Graders evaluate trajectory quality, not just final state.

Score range: (0.0, 1.0) per task — strictly between 0 and 1 exclusive.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.environment import ClinicalTrialEnvironment


# ── Score clamping (OpenEnv requires strictly (0, 1) exclusive) ──────────────

def _clamp_score(score: float) -> float:
    """Clamp score to the open interval (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, score))


# ── Generic utility checks ───────────────────────────────────────────────────

def _safety_score(env: "ClinicalTrialEnvironment") -> tuple[float, str]:
    """Check if safety was maintained throughout the episode."""
    if env.state.max_side_effects_seen <= 0.30:
        return 0.20, "PASS: Side effects kept within safe limits (≤30%) → +0.20"
    elif env.state.max_side_effects_seen <= 0.45:
        return 0.08, f"PARTIAL: Side effects peaked at {env.state.max_side_effects_seen:.1%} → +0.08"
    return 0.0, f"FAIL: Side effects reached {env.state.max_side_effects_seen:.1%} → +0.00"


def _effectiveness_score(env: "ClinicalTrialEnvironment") -> tuple[float, str]:
    """Check final drug effectiveness."""
    eff = env.effectiveness
    if eff >= 0.45:
        return 0.25, f"PASS: Strong effectiveness ({eff:.1%}) → +0.25"
    elif eff >= 0.30:
        return 0.15, f"PARTIAL: Moderate effectiveness ({eff:.1%}) → +0.15"
    elif eff >= 0.20:
        return 0.05, f"PARTIAL: Weak effectiveness ({eff:.1%}) → +0.05"
    return 0.0, f"FAIL: Drug not effective enough ({eff:.1%}) → +0.00"


def _budget_score(env: "ClinicalTrialEnvironment") -> tuple[float, str]:
    """Check budget management."""
    if env.budget <= 0:
        return 0.0, "FAIL: Budget fully exhausted → +0.00"
    pct_used = 1.0 - (env.budget / env.starting_budget)
    if pct_used < 0.50 and env.effectiveness >= 0.30:
        return 0.10, f"PASS: Efficient budget use ({pct_used:.0%} used) → +0.10"
    elif env.budget > 0:
        return 0.05, f"PARTIAL: Budget not exhausted ({pct_used:.0%} used) → +0.05"
    return 0.0, "FAIL: Budget exhausted → +0.00"


def _exploration_score(env: "ClinicalTrialEnvironment") -> tuple[float, str]:
    """Check whether agent explored dose space rather than staying at one dose."""
    doses = env.state.doses_visited
    unique_doses = len(set(doses))
    if unique_doses >= 4:
        return 0.15, f"PASS: Good exploration ({unique_doses} different doses tested) → +0.15"
    elif unique_doses >= 2:
        return 0.07, f"PARTIAL: Some exploration ({unique_doses} doses tested) → +0.07"
    return 0.0, f"FAIL: No exploration ({unique_doses} dose tested) → +0.00"


# ── Task-specific graders ────────────────────────────────────────────────────

def _grade_dose_escalation(env: "ClinicalTrialEnvironment") -> tuple[float, list[str]]:
    """
    Task 1: Classic dose-finding (easy)
    Find optimal dose through exploration, maintain safety, stop appropriately.
    """
    score = 0.0
    reasons = []

    # 1. Safety maintained (0.20)
    s, r = _safety_score(env)
    score += s; reasons.append(r)

    # 2. Effectiveness achieved (0.25)
    s, r = _effectiveness_score(env)
    score += s; reasons.append(r)

    # 3. Dose exploration (0.15)
    s, r = _exploration_score(env)
    score += s; reasons.append(r)

    # 4. Found dose close to optimal (0.20)
    dose_error = abs(env.current_dose - env.true_optimal_dose)
    if dose_error <= 10:
        score += 0.20
        reasons.append(f"PASS: Dose within 10mg of optimal ({dose_error}mg error) → +0.20")
    elif dose_error <= 25:
        score += 0.10
        reasons.append(f"PARTIAL: Dose within 25mg of optimal ({dose_error}mg error) → +0.10")
    else:
        reasons.append(f"FAIL: Dose {dose_error}mg from optimal → +0.00")

    # 5. Efficiency (0.10)
    if env.week <= 15 and env.effectiveness >= 0.30:
        score += 0.10
        reasons.append(f"BONUS: Found effective dose quickly (week {env.week}) → +0.10")

    # 6. Budget management (0.10)
    s, r = _budget_score(env)
    score += s; reasons.append(r)

    return _clamp_score(round(min(1.0, score), 3)), reasons


def _grade_adaptive_enrollment(env: "ClinicalTrialEnvironment") -> tuple[float, list[str]]:
    """
    Task 2: Adaptive enrollment (medium)
    Manage multiple treatment arms, add/drop arms based on data.
    """
    score = 0.0
    reasons = []

    # 1. Safety (0.15)
    s, r = _safety_score(env)
    score += s * 0.75; reasons.append(r)

    # 2. Effectiveness (0.20)
    s, r = _effectiveness_score(env)
    score += s * 0.80; reasons.append(r)

    # 3. Used adaptive features — arms added/dropped (0.20)
    arms_used = env.state.arms_added + env.state.arms_dropped
    if arms_used >= 3:
        score += 0.20
        reasons.append(f"PASS: Good adaptive management ({arms_used} arm changes) → +0.20")
    elif arms_used >= 1:
        score += 0.10
        reasons.append(f"PARTIAL: Some adaptation ({arms_used} arm changes) → +0.10")
    else:
        reasons.append("FAIL: No adaptive arm management used → +0.00")

    # 4. Exploration (0.15)
    s, r = _exploration_score(env)
    score += s; reasons.append(r)

    # 5. Patient efficiency (0.15)
    if env.patients_enrolled < 150 and env.effectiveness >= 0.30:
        score += 0.15
        reasons.append(f"PASS: Found result with {env.patients_enrolled} patients → +0.15")
    elif env.patients_enrolled < 250:
        score += 0.07
        reasons.append(f"PARTIAL: Used {env.patients_enrolled} patients → +0.07")
    else:
        reasons.append(f"FAIL: Too many patients ({env.patients_enrolled}) → +0.00")

    # 6. Budget (0.10)
    s, r = _budget_score(env)
    score += s; reasons.append(r)

    return _clamp_score(round(min(1.0, score), 3)), reasons


def _grade_interim_analysis(env: "ClinicalTrialEnvironment") -> tuple[float, list[str]]:
    """
    Task 3: Interim analysis (medium)
    Make proper continue/stop decisions at interim checkpoints.
    """
    score = 0.0
    reasons = []

    # 1. Safety (0.15)
    s, r = _safety_score(env)
    score += s * 0.75; reasons.append(r)

    # 2. Used interim analyses (0.25)
    interims = env.state.interim_analyses_requested
    if interims >= 2:
        score += 0.25
        reasons.append(f"PASS: Conducted {interims} interim analyses → +0.25")
    elif interims >= 1:
        score += 0.12
        reasons.append(f"PARTIAL: Conducted {interims} interim analysis → +0.12")
    else:
        reasons.append("FAIL: No interim analyses conducted → +0.00")

    # 3. Effectiveness (0.20)
    s, r = _effectiveness_score(env)
    score += s * 0.80; reasons.append(r)

    # 4. Statistical power achieved (0.15)
    if env.statistical_power >= 0.80:
        score += 0.15
        reasons.append(f"PASS: Adequate statistical power ({env.statistical_power:.1%}) → +0.15")
    elif env.statistical_power >= 0.60:
        score += 0.07
        reasons.append(f"PARTIAL: Moderate power ({env.statistical_power:.1%}) → +0.07")
    else:
        reasons.append(f"FAIL: Insufficient power ({env.statistical_power:.1%}) → +0.00")

    # 5. Appropriate trial length (0.15)
    if 10 <= env.week <= 30:
        score += 0.15
        reasons.append(f"PASS: Appropriate trial duration ({env.week} weeks) → +0.15")
    elif env.week < 10:
        score += 0.05
        reasons.append(f"PARTIAL: Trial too short ({env.week} weeks) → +0.05")
    else:
        reasons.append(f"FAIL: Trial too long ({env.week} weeks) → +0.00")

    # 6. Budget (0.10)
    s, r = _budget_score(env)
    score += s; reasons.append(r)

    return _clamp_score(round(min(1.0, score), 3)), reasons


def _grade_safety_monitoring(env: "ClinicalTrialEnvironment") -> tuple[float, list[str]]:
    """
    Task 4: Safety monitoring (hard)
    Detect emerging safety signals, respond appropriately.
    """
    score = 0.0
    reasons = []

    # 1. Detected safety concerns — paused enrollment or stopped (0.25)
    if env.state.enrollment_paused or (
        env.trial_stopped and env.side_effect_rate > 0.20
    ):
        score += 0.25
        reasons.append("PASS: Responded to safety signals (paused/stopped) → +0.25")
    elif env.state.safety_responses >= 2:
        score += 0.15
        reasons.append(f"PARTIAL: Made {env.state.safety_responses} safety responses → +0.15")
    else:
        reasons.append("FAIL: Did not adequately respond to safety signals → +0.00")

    # 2. Monitored organ toxicity (0.20)
    if env.state.safety_responses >= 3:
        score += 0.20
        reasons.append(f"PASS: Active safety monitoring ({env.state.safety_responses} responses) → +0.20")
    elif env.state.safety_responses >= 1:
        score += 0.10
        reasons.append(f"PARTIAL: Some monitoring ({env.state.safety_responses} responses) → +0.10")
    else:
        reasons.append("FAIL: No proactive safety monitoring → +0.00")

    # 3. Limited SAEs (0.15)
    if env.state.sae_count <= 1:
        score += 0.15
        reasons.append(f"PASS: Minimal SAEs ({env.state.sae_count}) → +0.15")
    elif env.state.sae_count <= 3:
        score += 0.07
        reasons.append(f"PARTIAL: Some SAEs ({env.state.sae_count}) → +0.07")
    else:
        reasons.append(f"FAIL: Too many SAEs ({env.state.sae_count}) → +0.00")

    # 4. Overall safety (0.15)
    s, r = _safety_score(env)
    score += s * 0.75; reasons.append(r)

    # 5. Still found some effectiveness (0.15)
    if env.effectiveness >= 0.25:
        score += 0.15
        reasons.append(f"PASS: Maintained effectiveness ({env.effectiveness:.1%}) despite safety focus → +0.15")
    else:
        reasons.append(f"FAIL: Drug ineffective ({env.effectiveness:.1%}) → +0.00")

    # 6. Budget (0.10)
    s, r = _budget_score(env)
    score += s; reasons.append(r)

    return _clamp_score(round(min(1.0, score), 3)), reasons


def _grade_multi_endpoint(env: "ClinicalTrialEnvironment") -> tuple[float, list[str]]:
    """
    Task 5: Multi-endpoint optimization (hard)
    Balance primary AND secondary endpoints simultaneously.
    """
    score = 0.0
    reasons = []

    # 1. Primary endpoint (0.20)
    s, r = _effectiveness_score(env)
    score += s * 0.80; reasons.append(r.replace("→", "(primary) →"))

    # 2. Secondary endpoint (0.20)
    sec = env.secondary_effectiveness
    if sec >= 0.35:
        score += 0.20
        reasons.append(f"PASS: Strong secondary endpoint ({sec:.1%}) → +0.20")
    elif sec >= 0.20:
        score += 0.10
        reasons.append(f"PARTIAL: Moderate secondary endpoint ({sec:.1%}) → +0.10")
    else:
        reasons.append(f"FAIL: Weak secondary endpoint ({sec:.1%}) → +0.00")

    # 3. Safety (0.15)
    s, r = _safety_score(env)
    score += s * 0.75; reasons.append(r)

    # 4. Exploration across dose space (0.15)
    s, r = _exploration_score(env)
    score += s; reasons.append(r)

    # 5. Trade-off management — both endpoints above threshold (0.15)
    if env.effectiveness >= 0.30 and sec >= 0.25:
        score += 0.15
        reasons.append("PASS: Both endpoints above threshold → +0.15")
    elif env.effectiveness >= 0.20 or sec >= 0.20:
        score += 0.05
        reasons.append("PARTIAL: Only one endpoint above threshold → +0.05")
    else:
        reasons.append("FAIL: Neither endpoint above threshold → +0.00")

    # 6. Budget (0.10)
    s, r = _budget_score(env)
    score += s; reasons.append(r)

    return _clamp_score(round(min(1.0, score), 3)), reasons


# ── Public API ────────────────────────────────────────────────────────────────

GRADERS = {
    "dose_escalation": _grade_dose_escalation,
    "adaptive_enrollment": _grade_adaptive_enrollment,
    "interim_analysis": _grade_interim_analysis,
    "safety_monitoring": _grade_safety_monitoring,
    "multi_endpoint": _grade_multi_endpoint,
}


def grade_episode(environment: "ClinicalTrialEnvironment", task_name: str = "") -> tuple[float, list[str]]:
    """
    Called once at the end of an episode.
    Returns (score: float in 0.0–1.0, reasons: list[str]).
    Dispatches to the task-specific grading rubric.
    """
    grader_fn = GRADERS.get(task_name)
    if grader_fn:
        return grader_fn(environment)

    # Fallback: use dose_escalation grader for unknown tasks
    return _grade_dose_escalation(environment)


def grade_by_task(environment: "ClinicalTrialEnvironment", task_name: str) -> tuple[float, list[str]]:
    """
    Task-aware wrapper. Each task uses its own rubric.
    No difficulty multipliers — difficulty is built into each rubric.
    Final score is clamped to (0, 1) exclusive per OpenEnv spec.
    """
    score, reasons = grade_episode(environment, task_name)
    config = {}
    try:
        from server.environment import TASK_CONFIG
        config = TASK_CONFIG.get(task_name, {})
    except ImportError:
        pass
    difficulty = config.get("difficulty", "easy")
    score = _clamp_score(round(score, 3))
    reasons.append(f"[{task_name}] difficulty={difficulty} final_score={score:.3f}")
    return score, reasons