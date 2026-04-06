"""
Clinical Trial Optimization — Core Environment Logic
Implements the OpenEnv Environment base class with typed generics.
"""
import random
import uuid
from typing import Optional

from openenv.core.env_server import Environment

# Import sibling models — works whether invoked from project root or server/
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import TrialAction, TrialObservation, TrialState


class ClinicalTrialEnvironment(Environment[TrialAction, TrialObservation, TrialState]):
    """
    A Phase-2 clinical drug-trial simulation.

    The agent must discover the hidden optimal dose through sequential
    decisions about dosage adjustment, patient enrollment, and trial
    termination — while managing safety and budget constraints.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # Stateless per HTTP call

    def __init__(
        self,
        difficulty: str = "easy",
        transform=None,
        rubric=None,
    ):
        super().__init__(transform=transform, rubric=rubric)
        self.difficulty = difficulty

        # These get set fresh each episode in reset()
        self._current_state: Optional[TrialState] = None
        self.week = 0
        self.current_dose = 0
        self.patients_enrolled = 0
        self.budget = 0.0
        self.side_effect_rate = 0.0
        self.effectiveness = 0.0
        self.true_optimal_dose = 0
        self.safe_dose_limit = 0
        self.trial_stopped = False

    # ─────────────────────────────────────────
    # RESET — called at the start of each episode
    # Signature matches openenv.core.env_server.Environment.reset()
    # ─────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> TrialObservation:
        self._reset_rubric()

        if seed is not None:
            random.seed(seed)

        ep_id = episode_id or str(uuid.uuid4())
        self.week = 1
        self.current_dose = 10          # Always start low for safety
        self.patients_enrolled = 20     # Start with 20 patients
        self.budget = 5_000_000.0       # $5 million budget
        self.trial_stopped = False

        # Set difficulty-specific hidden truth
        # (also accept difficulty override via kwargs for per-request control)
        difficulty = kwargs.get("difficulty", self.difficulty)

        if difficulty == "hard":
            # Narrow window: 60-100mg optimal, only 10mg safety margin
            self.true_optimal_dose = random.randint(60, 100)
            self.safe_dose_limit = self.true_optimal_dose + random.randint(8, 12)
        elif difficulty == "medium":
            # Moderate: 50-80mg optimal, 15mg safety margin
            self.true_optimal_dose = random.randint(50, 80)
            self.safe_dose_limit = self.true_optimal_dose + random.randint(13, 18)
        else:
            # Easy: 30-70mg, generous 20-35mg safety margin
            self.true_optimal_dose = random.randint(30, 70)
            self.safe_dose_limit = self.true_optimal_dose + random.randint(20, 35)

        # Calculate initial readings at starting dose
        self.effectiveness = self._simulate_effectiveness(self.current_dose)
        self.side_effect_rate = self._simulate_side_effects(self.current_dose)

        self._current_state = TrialState(
            episode_id=ep_id,
            step_count=0,
            task_name=kwargs.get("task_name", ""),
            difficulty=difficulty,
            true_optimal_dose=self.true_optimal_dose,
            safe_dose_limit=self.safe_dose_limit,
        )

        return self._make_observation(
            reward=0.0,
            message="Trial started. Initial dose: 10mg on 20 patients.",
        )

    # ─────────────────────────────────────────
    # STEP — called every time AI makes a decision
    # Signature matches openenv.core.env_server.Environment.step()
    # ─────────────────────────────────────────
    def step(
        self,
        action: TrialAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> TrialObservation:
        if self._current_state is None:
            # Auto-reset if called before reset()
            self.reset()

        self._current_state.step_count += 1
        reward = 0.0
        message = ""

        # ── PROCESS THE AI'S DECISION ──

        if action.decision == "increase_dose":
            old_dose = self.current_dose
            self.current_dose = min(self.current_dose + 20, 150)
            self.budget -= 150_000
            message = f"Increased dose from {old_dose}mg to {self.current_dose}mg."

            self.effectiveness = self._simulate_effectiveness(self.current_dose)
            self.side_effect_rate = self._simulate_side_effects(self.current_dose)

            if self.side_effect_rate > 0.30:
                reward = -0.30   # Normalised: entered unsafe zone
                message += " WARNING: High side effects observed!"
            elif self.effectiveness > 0.40:
                reward = +0.25   # Found a very effective dose
                message += " Good result — strong effectiveness."
            elif self.effectiveness > 0.25:
                reward = +0.10   # Moving in right direction
                message += " Moderate improvement in effectiveness."
            else:
                reward = -0.05   # Went higher but no improvement
                message += " No improvement — may have overshot."

        elif action.decision == "decrease_dose":
            old_dose = self.current_dose
            self.current_dose = max(self.current_dose - 15, 5)
            self.budget -= 100_000
            message = f"Decreased dose from {old_dose}mg to {self.current_dose}mg."

            self.effectiveness = self._simulate_effectiveness(self.current_dose)
            self.side_effect_rate = self._simulate_side_effects(self.current_dose)

            if self.side_effect_rate < 0.15 and self.effectiveness > 0.30:
                reward = +0.15   # Smart safety move while keeping efficacy
                message += " Good balance of safety and effectiveness."
            elif self.effectiveness < 0.15:
                reward = -0.10   # Went too low — lost effectiveness
                message += " Effectiveness dropped too low."
            else:
                reward = +0.05

        elif action.decision == "keep_dose":
            self.patients_enrolled += 20
            self.budget -= 80_000
            message = f"Kept dose at {self.current_dose}mg. Enrolled 20 more patients."

            if self.patients_enrolled < 100:
                reward = +0.05   # Gathering data — makes sense
            else:
                reward = -0.08   # Wasting money — already have enough data
                message += " Warning: already have sufficient patient data."

        elif action.decision == "enroll_more_patients":
            self.patients_enrolled += 40
            self.budget -= 200_000
            message = f"Enrolled 40 additional patients at {self.current_dose}mg."

            if self.patients_enrolled < 150:
                reward = +0.08
            else:
                reward = -0.10   # Wasteful at this point
                message += " Unnecessary — trial already well-powered."

        elif action.decision == "stop_trial":
            self.trial_stopped = True
            message = "Trial stopped by agent."

            if self.effectiveness > 0.35 and self.side_effect_rate < 0.20:
                reward = +0.50   # Perfect — found good dose, stopped cleanly
                message += " Excellent decision — effective and safe dose confirmed."
            elif self.effectiveness < 0.15:
                reward = +0.15   # Right to stop a failing trial
                message += " Correct to stop — drug not effective enough."
            elif self.side_effect_rate > 0.30:
                reward = +0.20   # Right to stop — safety issue
                message += " Correct to stop — safety threshold exceeded."
            else:
                reward = -0.15   # Stopped too early without clear reason
                message += " Premature stop — more data was needed."

        else:
            reward = -0.05
            message = f"Unknown action: {action.decision}"

        # ── ADVANCE TIME ──
        self.week += 1
        self.budget -= 50_000   # Weekly running costs regardless of action

        # ── CHECK IF EPISODE IS DONE ──
        done = (
            self.trial_stopped
            or self.week > 52
            or self.budget <= 0
            or self.side_effect_rate > 0.50
        )

        if self.budget <= 0 and not self.trial_stopped:
            reward -= 0.20
            message += " TRIAL ENDED: Budget exhausted."
        if self.side_effect_rate > 0.50:
            reward -= 0.40
            message += " EMERGENCY STOP: Dangerous side effect rate."

        # Clamp reward to reasonable range
        reward = round(max(-1.0, min(1.0, reward)), 4)

        return self._make_observation(reward=reward, message=message, done=done)

    # ─────────────────────────────────────────
    # STATE — metadata about the episode
    # Decorated as property to match abstract base
    # ─────────────────────────────────────────
    @property
    def state(self) -> TrialState:
        if self._current_state is None:
            # Return a default state if reset() hasn't been called
            return TrialState()
        return self._current_state

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────
    def _simulate_effectiveness(self, dose: int) -> float:
        """
        Simulates how effective the drug is at a given dose.
        Peaks near true_optimal_dose, drops off on both sides.
        The AI cannot see this formula — it must discover the shape
        through experimentation.
        """
        distance = abs(dose - self.true_optimal_dose)
        effectiveness = max(0.0, 1.0 - (distance / 50.0))
        noise = random.uniform(-0.05, 0.05)
        return round(min(1.0, max(0.0, effectiveness + noise)), 3)

    def _simulate_side_effects(self, dose: int) -> float:
        """
        Side effects increase as dose exceeds the safe limit.
        Below safe_dose_limit: minimal side effects.
        Above: rapidly increasing danger.
        """
        if dose <= self.safe_dose_limit:
            base = dose / (self.safe_dose_limit * 5)
        else:
            excess = dose - self.safe_dose_limit
            base = 0.15 + (excess / 80.0)

        noise = random.uniform(-0.03, 0.03)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _make_observation(
        self,
        reward: float,
        message: str,
        done: bool = False,
    ) -> TrialObservation:
        return TrialObservation(
            week=self.week,
            current_dose_mg=self.current_dose,
            patients_enrolled=self.patients_enrolled,
            avg_effectiveness=self.effectiveness,
            side_effect_rate=self.side_effect_rate,
            budget_remaining=self.budget,
            done=done,
            reward=reward,
            message=message,
        )