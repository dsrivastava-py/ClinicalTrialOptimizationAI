"""
Clinical Trial Optimization — Baseline Inference Script
Uses the OpenAI client to run an LLM agent against all 3 tasks.

MANDATORY ENVIRONMENT VARIABLES:
  HF_TOKEN       — API key (Groq: gsk_..., HuggingFace: hf_...)
  API_BASE_URL   — LLM inference endpoint
  MODEL_NAME     — Model identifier to use

MANDATORY LOG FORMAT:
  [START] task=<name> env=<env> model=<model>
  [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<null|msg>
  [END]   success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
"""
import os
import random
import sys
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import ClinicalTrialEnvironment
from server.grader import grade_by_task
from models import TrialAction

# ── MANDATORY ENV VARIABLES ──────────────────────────────────────────────────
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

BENCHMARK              = "clinical_trial_env"
MAX_STEPS              = 20
SUCCESS_SCORE_THRESHOLD = 0.5
REPRODUCIBILITY_SEED   = 42   # Fixed seed — ensures same scores every run

VALID_ACTIONS = [
    "increase_dose",
    "decrease_dose",
    "keep_dose",
    "enroll_more_patients",
    "stop_trial",
]

TASKS = [
    {"name": "dose_finding_easy",   "difficulty": "easy"},
    {"name": "dose_finding_medium", "difficulty": "medium"},
    {"name": "dose_finding_hard",   "difficulty": "hard"},
]


# ── MANDATORY STDOUT FORMAT ───────────────────────────────────────────────────

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


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM DECISION MAKER ───────────────────────────────────────────────────────

def ask_llm(client: OpenAI, observation, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"

    prompt = f"""You are managing a clinical drug trial for Type 2 Diabetes.

Current Trial Status:
- Week: {observation.week} of 52
- Current dose: {observation.current_dose_mg}mg
- Patients enrolled: {observation.patients_enrolled}
- Drug effectiveness: {observation.avg_effectiveness:.1%} average blood sugar reduction
- Side effect rate: {observation.side_effect_rate:.1%} of patients affected
- Budget remaining: ${observation.budget_remaining:,.0f}
- Last outcome: {observation.message}

Recent decisions:
{history_block}

PRIORITY RULES (apply the FIRST matching rule — stop there):
1. SAFETY: If side_effect_rate > 25% → decrease_dose
2. SUCCESS: If effectiveness > 35% AND side_effect_rate < 20% → stop_trial
3. EXPLORE UP: If effectiveness < 30% AND side_effect_rate < 20% → increase_dose
4. SAFE LOCK-IN: If effectiveness >= 30% AND patients < 80 AND side_effect_rate < 20% → keep_dose
5. STOP WHEN POWERED: If effectiveness >= 30% AND patients >= 80 → stop_trial
6. DANGEROUS: If effectiveness < 20% AND side_effect_rate > 20% → decrease_dose
7. DEFAULT: increase_dose

IMPORTANT: The trial starts at 10mg. The optimal dose is MUCH HIGHER (30-100mg).
If effectiveness is below 30% and side effects are safe, you MUST choose increase_dose.

Choose EXACTLY ONE action:
increase_dose, decrease_dose, keep_dose, enroll_more_patients, stop_trial

Reply with ONLY the action name. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower()
        for action in VALID_ACTIONS:
            if action in raw:
                decision = action
                break
        else:
            decision = "increase_dose"

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        decision = "increase_dose"

    # ── SAFETY OVERRIDE ──────────────────────────────────────────────────────
    # Prevent LLM from getting stuck on keep_dose when the dose is clearly
    # too low to be effective. This is the single most common LLM failure mode.
    if decision == "keep_dose" and observation.avg_effectiveness < 0.25:
        decision = "increase_dose"

    # Prevent stop_trial when effectiveness is too low — wasteful termination
    if decision == "stop_trial" and observation.avg_effectiveness < 0.20:
        decision = "increase_dose"

    return decision


# ── SINGLE EPISODE ────────────────────────────────────────────────────────────

def run_episode(task_name: str, difficulty: str) -> float:
    # Fix random seed for REPRODUCIBILITY — same seed = same hidden optimal
    # dose = same scores every run. Required by hackathon spec.
    random.seed(REPRODUCIBILITY_SEED)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Difficulty is baked into the constructor — reset() reads self.difficulty
    env = ClinicalTrialEnvironment(difficulty=difficulty)

    # Pass seed so reset() also uses the fixed seed for its random.seed() call
    obs = env.reset(task_name=task_name, seed=REPRODUCIBILITY_SEED)

    history:    List[str]   = []
    rewards:    List[float] = []
    steps_taken = 0
    success     = False
    score       = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            decision = ask_llm(client, obs, history)
            action   = TrialAction(decision=decision)
            obs      = env.step(action)

            reward = float(obs.reward) if obs.reward is not None else 0.0
            done   = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=decision, reward=reward,
                     done=done, error=None)

            history.append(
                f"Step {step}: {decision} → "
                f"dose={obs.current_dose_mg}mg "
                f"effect={obs.avg_effectiveness:.0%} "
                f"se={obs.side_effect_rate:.0%} "
                f"reward={reward:+.2f}"
            )

            if done:
                break

        score, reasons = grade_by_task(env, task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    print(f"[GRADE] task={task_name} score={score:.3f}", flush=True)
    for r in reasons:
        prefix = "✅" if ("PASS" in r or "BONUS" in r) else "❌" if "FAIL" in r else "ℹ️"
        print(f"[GRADE] {prefix} {r}", flush=True)

    return score


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("CLINICAL TRIAL ENV — Baseline Evaluation", flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(f"Endpoint: {API_BASE_URL}", flush=True)
    print(f"Seed:     {REPRODUCIBILITY_SEED} (fixed for reproducibility)", flush=True)
    print("=" * 60, flush=True)

    if not API_KEY:
        print(
            "[ERROR] No API key found. "
            "Set HF_TOKEN, OPENAI_API_KEY, or API_KEY environment variable.",
            flush=True,
        )
        sys.exit(1)

    scores = {}
    for task in TASKS:
        print(f"\n{'─'*60}", flush=True)
        print(f"Task: {task['name'].upper()}", flush=True)
        print(f"{'─'*60}", flush=True)
        score = run_episode(
            task_name=task["name"],
            difficulty=task["difficulty"],
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