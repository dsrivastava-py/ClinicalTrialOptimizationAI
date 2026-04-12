"""
Clinical Trial Optimization — Baseline Inference Script
Uses the OpenAI client to run an LLM agent against all 5 tasks.

MANDATORY ENVIRONMENT VARIABLES:
  HF_TOKEN       — Hugging Face API token (required, no default)
  API_BASE_URL   — LLM inference endpoint (default: HuggingFace router)
  MODEL_NAME     — Model identifier (default: Qwen/Qwen2.5-7B-Instruct)

MANDATORY LOG FORMAT (OpenEnv spec):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""
import os
import random
import sys
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import ClinicalTrialEnvironment, TASK_CONFIG, VALID_ACTIONS
from server.grader import grade_by_task
from models import TrialAction

# ── MANDATORY ENV VARIABLES ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK              = "clinical_trial_env"
SUCCESS_SCORE_THRESHOLD = 0.5
REPRODUCIBILITY_SEED   = 42


# ── TASK DEFINITIONS ─────────────────────────────────────────────────────────
TASKS = [
    {"name": "dose_escalation",     "max_steps": 20},
    {"name": "adaptive_enrollment", "max_steps": 25},
    {"name": "interim_analysis",    "max_steps": 30},
    {"name": "safety_monitoring",   "max_steps": 25},
    {"name": "multi_endpoint",      "max_steps": 30},
]


# ── MANDATORY STDOUT FORMAT (OpenEnv spec) ────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── LLM DECISION MAKER ──────────────────────────────────────────────────────

def ask_llm(client: OpenAI, obs, history: List[str], task_name: str) -> str:
    """Ask the LLM to choose an action based on the current observation."""
    history_block = "\n".join(history[-5:]) if history else "None"

    # Build task-specific context
    task_context = _get_task_context(task_name)

    # Build organ toxicity string
    tox_str = ", ".join(
        f"{organ}={level:.1%}" for organ, level in obs.organ_toxicity.items()
    )

    prompt = f"""You are an AI clinical trial manager running a Phase 2 drug trial.

TASK: {task_name}
{task_context}

CURRENT TRIAL STATUS (Week {obs.week}/52):
  Dose: {obs.current_dose_mg}mg
  Patients: {obs.patients_enrolled}
  Effectiveness: {obs.avg_effectiveness:.1%} (95% CI: {obs.confidence_interval_low:.1%}–{obs.confidence_interval_high:.1%})
  Side effects: {obs.side_effect_rate:.1%}
  Statistical power: {obs.statistical_power:.1%}
  Organ toxicity: {tox_str}
  SAEs: {obs.serious_adverse_events}
  Dropout rate: {obs.dropout_rate:.1%}
  Treatment arms: {obs.active_treatment_arms}
  Budget remaining: ${obs.budget_remaining:,.0f}
  Futility probability: {obs.futility_probability:.1%}
  Interim analysis due: {obs.interim_analysis_due}
  Data maturity: {obs.data_maturity:.1%}
  Last outcome: {obs.message}

RECENT DECISIONS:
{history_block}

DECISION RULES:
1. SAFETY FIRST: If side_effect_rate > 25% or any organ toxicity > 25% → decrease_dose or pause_enrollment
2. EXPLORE: If effectiveness < 30% and side effects < 20% → increase_dose
3. INTERIM: If interim_analysis_due or want to check futility → request_interim_analysis
4. ADAPTIVE: For adaptive_enrollment, manage arms with add_treatment_arm / drop_treatment_arm
5. STOP: If effectiveness > 35% and side effects < 20% and statistical_power > 70% → stop_trial
6. MONITOR: For safety_monitoring, use adjust_monitoring when toxicity emerging
7. DEFAULT: increase_dose (start low, explore upward)

IMPORTANT: The trial starts at 10mg. The optimal dose is MUCH HIGHER (30-100mg).
You must EXPLORE by increasing the dose. Staying at 10mg will achieve nothing.

Available actions: {', '.join(VALID_ACTIONS)}

Reply with ONLY the action name. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower()
        for action in VALID_ACTIONS:
            if action in raw:
                return action
        return "increase_dose"  # Default: explore upward

    except Exception:
        return "increase_dose"


def _get_task_context(task_name: str) -> str:
    """Return task-specific instructions for the LLM prompt."""
    contexts = {
        "dose_escalation": (
            "Find the optimal dose using 3+3 escalation. Increase dose systematically, "
            "watch for side effects, and stop when you find an effective safe dose."
        ),
        "adaptive_enrollment": (
            "Manage multiple treatment arms. Add arms at promising doses, drop "
            "underperforming arms. Use add_treatment_arm and drop_treatment_arm."
        ),
        "interim_analysis": (
            "Make continue/stop decisions at interim analysis points. Use "
            "request_interim_analysis to check futility and statistical power."
        ),
        "safety_monitoring": (
            "Watch for emerging organ toxicity signals. Use adjust_monitoring, "
            "pause_enrollment, and decrease_dose when safety signals appear."
        ),
        "multi_endpoint": (
            "Optimize for BOTH primary and secondary endpoints. The secondary "
            "endpoint peaks at a LOWER dose than the primary — find the sweet spot."
        ),
    }
    return contexts.get(task_name, "Find the optimal dose.")


# ── SAFETY OVERRIDES ─────────────────────────────────────────────────────────

def apply_safety_overrides(decision: str, obs, task_name: str) -> str:
    """Hard-coded safety overrides for common LLM failure modes."""
    # Prevent keep_dose at an ineffective dose (most common LLM failure)
    if decision == "keep_dose" and obs.avg_effectiveness < 0.20:
        return "increase_dose"

    # Prevent premature stop_trial
    if decision == "stop_trial" and obs.avg_effectiveness < 0.15:
        return "increase_dose"

    # Force safety response when side effects are critical
    if obs.side_effect_rate > 0.30 and decision not in ("decrease_dose", "pause_enrollment", "stop_trial"):
        return "decrease_dose"

    # For safety_monitoring: force monitoring when organ toxicity is high
    if task_name == "safety_monitoring":
        max_tox = max(obs.organ_toxicity.values()) if obs.organ_toxicity else 0
        if max_tox > 0.25 and decision not in ("decrease_dose", "pause_enrollment", "adjust_monitoring"):
            return "pause_enrollment"

    return decision


# ── SINGLE EPISODE ────────────────────────────────────────────────────────────

def run_episode(task_name: str, max_steps: int) -> float:
    """Run a single episode for the given task. Returns grader score."""
    random.seed(REPRODUCIBILITY_SEED)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    config = TASK_CONFIG.get(task_name, TASK_CONFIG["dose_escalation"])
    env = ClinicalTrialEnvironment(difficulty=config["difficulty"])
    obs = env.reset(task_name=task_name, seed=REPRODUCIBILITY_SEED)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken = 0
    success     = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            if obs.done:
                break

            decision = ask_llm(client, obs, history, task_name)
            decision = apply_safety_overrides(decision, obs, task_name)

            action = TrialAction(decision=decision)
            obs    = env.step(action)

            reward = float(obs.reward) if obs.reward is not None else 0.0
            done   = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=decision, reward=reward,
                     done=done, error=last_error)

            history.append(
                f"Step {step}: {decision} → "
                f"dose={obs.current_dose_mg}mg "
                f"effect={obs.avg_effectiveness:.0%} "
                f"se={obs.side_effect_rate:.0%} "
                f"power={obs.statistical_power:.0%} "
                f"reward={reward:+.2f}"
            )

            if done:
                break

        score, reasons = grade_by_task(env, task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        last_error = str(e)
        score = 0.0
        reasons = [f"ERROR: {last_error}"]

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    # Print grading details (informational, not part of spec format)
    print(f"[GRADE] task={task_name} score={score:.3f}", flush=True)
    for r in reasons:
        prefix = "✅" if "PASS" in r else "❌" if "FAIL" in r else "ℹ️"
        print(f"[GRADE] {prefix} {r}", flush=True)

    return score


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("CLINICAL TRIAL ENV — Baseline Evaluation", flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(f"Endpoint: {API_BASE_URL}", flush=True)
    print(f"Tasks:    {len(TASKS)}", flush=True)
    print(f"Seed:     {REPRODUCIBILITY_SEED} (fixed for reproducibility)", flush=True)
    print("=" * 60, flush=True)

    scores = {}
    for task in TASKS:
        print(f"\n{'─'*60}", flush=True)
        print(f"Task: {task['name'].upper()}", flush=True)
        print(f"{'─'*60}", flush=True)
        score = run_episode(
            task_name=task["name"],
            max_steps=task["max_steps"],
        )
        scores[task["name"]] = score

    print(f"\n{'='*60}", flush=True)
    print("FINAL BASELINE SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for task_name, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_name:32s}: {score:.3f}  {bar}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':32s}: {avg:.3f}", flush=True)
    print(f"{'='*60}", flush=True)