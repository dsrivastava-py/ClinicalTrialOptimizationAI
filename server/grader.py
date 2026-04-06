"""
Per-task grader: returns a score in [0.0, 1.0] with partial-credit sub-scores.
Each grader criterion is independent and clearly deterministic.
"""
from __future__ import annotations


def grade_episode(environment, task_name: str = "") -> tuple[float, list[str]]:
    """
    Called once at the end of an episode.
    Returns (score: float in 0.0–1.0, reasons: list[str]).

    Scoring rubric (max 1.0):
      Safety maintained (side_effect_rate < 0.30)       → 0.25
      Effectiveness ≥ 0.40 (strong)                     → 0.30
      Effectiveness ≥ 0.25 but < 0.40 (moderate)        → 0.15
      Efficiency  (found dose in ≤ 20 weeks)             → 0.20
      Budget managed (< 50% used AND effectiveness good) → 0.15
      Avoided budget exhaustion                          → 0.10
    """
    score = 0.0
    reasons: list[str] = []

    # ── SAFETY CHECK (highest weight, most important) ──
    if environment.side_effect_rate <= 0.30:
        score += 0.25
        reasons.append("PASS: Side effects kept within safe limits (≤30%) → +0.25")
    else:
        reasons.append(
            f"FAIL: Side effects exceeded 30% threshold "
            f"({environment.side_effect_rate:.1%}) → +0.00"
        )

    # ── EFFECTIVENESS CHECK ──
    if environment.effectiveness >= 0.40:
        score += 0.30
        reasons.append("PASS: Strong effectiveness achieved (≥40%) → +0.30")
    elif environment.effectiveness >= 0.25:
        score += 0.15
        reasons.append("PARTIAL: Moderate effectiveness (25-40%) → +0.15")
    else:
        reasons.append(
            f"FAIL: Drug not effective enough ({environment.effectiveness:.1%}) → +0.00"
        )

    # ── EFFICIENCY CHECK (found result without wasting weeks) ──
    if environment.week <= 20 and environment.effectiveness >= 0.35:
        score += 0.20
        reasons.append("BONUS: Found effective dose quickly (≤20 weeks) → +0.20")
    elif environment.week > 45:
        reasons.append("PENALTY: Trial ran too long without conclusion → +0.00")

    # ── BUDGET CHECK ──
    starting_budget = 5_000_000.0
    budget_used_pct = 1.0 - (environment.budget / starting_budget)

    if environment.budget > 0:
        score += 0.10
        reasons.append("PASS: Budget not exhausted → +0.10")
    else:
        reasons.append("FAIL: Budget fully exhausted → +0.00")

    if budget_used_pct < 0.50 and environment.effectiveness >= 0.35:
        score += 0.15
        reasons.append("BONUS: Achieved result using <50% of budget → +0.15")

    # ── CLAMP TO 0.0–1.0 ──
    final_score = round(max(0.0, min(1.0, score)), 3)
    return final_score, reasons


def grade_by_task(environment, task_name: str) -> tuple[float, list[str]]:
    """
    Task-aware wrapper that applies difficulty penalty multipliers.
    Easy tasks have more lenient scoring; hard tasks apply a difficulty bonus.
    """
    score, reasons = grade_episode(environment, task_name)

    if "hard" in task_name:
        # On hard tasks, reward better if any partial success
        if score > 0.0:
            score = min(1.0, score * 0.95)   # Slight normalisation, hard is expected harder
        reasons.append(f"[Hard task] Final adjusted score: {score:.3f}")
    elif "medium" in task_name:
        reasons.append(f"[Medium task] Final score: {score:.3f}")
    else:
        reasons.append(f"[Easy task] Final score: {score:.3f}")

    return round(score, 3), reasons