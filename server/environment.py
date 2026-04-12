"""
Clinical Trial Optimization — Core Environment Logic
Implements the OpenEnv Environment base class with 5 distinct clinical
trial management tasks, rich simulation, and trajectory tracking.

Tasks:
  1. dose_escalation     — Classic 3+3 dose-finding
  2. adaptive_enrollment — Bayesian adaptive patient randomization
  3. interim_analysis    — Futility/efficacy stopping decisions
  4. safety_monitoring   — DSMB-style safety pattern recognition
  5. multi_endpoint      — Multi-objective primary+secondary optimization
"""
import math
import random
import uuid
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import TrialAction, TrialObservation, TrialState


# ── Task configurations ──────────────────────────────────────────────────────
TASK_CONFIG = {
    "dose_escalation": {
        "difficulty": "easy",
        "description": "Find the optimal dose via 3+3 escalation design",
        "max_steps": 20,
        "optimal_dose_range": (30, 70),
        "safety_margin_range": (20, 35),
        "starting_budget": 5_000_000,
    },
    "adaptive_enrollment": {
        "difficulty": "medium",
        "description": "Manage patient enrollment across treatment arms",
        "max_steps": 25,
        "optimal_dose_range": (40, 80),
        "safety_margin_range": (15, 25),
        "starting_budget": 8_000_000,
    },
    "interim_analysis": {
        "difficulty": "medium",
        "description": "Make continue/stop decisions at interim analysis points",
        "max_steps": 30,
        "optimal_dose_range": (50, 90),
        "safety_margin_range": (12, 20),
        "starting_budget": 7_000_000,
    },
    "safety_monitoring": {
        "difficulty": "hard",
        "description": "Detect and respond to emerging multi-organ safety signals",
        "max_steps": 25,
        "optimal_dose_range": (60, 100),
        "safety_margin_range": (8, 14),
        "starting_budget": 6_000_000,
    },
    "multi_endpoint": {
        "difficulty": "hard",
        "description": "Optimize primary AND secondary endpoints simultaneously",
        "max_steps": 30,
        "optimal_dose_range": (40, 90),
        "safety_margin_range": (10, 18),
        "starting_budget": 9_000_000,
    },
}

VALID_ACTIONS = [
    "increase_dose",
    "decrease_dose",
    "keep_dose",
    "enroll_more_patients",
    "stop_trial",
    "add_treatment_arm",
    "drop_treatment_arm",
    "request_interim_analysis",
    "pause_enrollment",
    "adjust_monitoring",
]


class ClinicalTrialEnvironment(Environment[TrialAction, TrialObservation, TrialState]):
    """
    A Phase-2 clinical drug-trial simulation with 5 distinct task types.

    The agent must discover the hidden optimal dose through sequential
    decisions while managing safety, patient ethics, statistical power,
    and budget constraints. Each task emphasizes a different aspect of
    real-world clinical trial management.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        difficulty: str = "easy",
        transform=None,
        rubric=None,
    ):
        super().__init__(transform=transform, rubric=rubric)
        self.difficulty = difficulty
        self._current_state: Optional[TrialState] = None
        self._init_episode_vars()

    def _init_episode_vars(self):
        """Initialize all episode-level variables."""
        self.week = 0
        self.current_dose = 0
        self.patients_enrolled = 0
        self.budget = 0.0
        self.starting_budget = 5_000_000.0
        self.side_effect_rate = 0.0
        self.effectiveness = 0.0
        self.true_optimal_dose = 0
        self.safe_dose_limit = 0
        self.trial_stopped = False
        self.enrollment_paused = False

        # Patient dynamics
        self.dropout_rate = 0.0
        self.treatment_arms: List[Dict] = []  # [{dose, patients, effectiveness}]
        self.serious_adverse_events = 0

        # Organ toxicity tracking
        self.organ_toxicity: Dict[str, float] = {
            "liver": 0.0,
            "kidney": 0.0,
            "cardiac": 0.0,
        }

        # Statistical tracking
        self.confidence_interval_width = 0.5
        self.statistical_power = 0.0

        # Interim analysis
        self.interim_checkpoints = []  # weeks at which interims are due
        self.interims_completed = 0

        # Task-specific state
        self.secondary_effectiveness = 0.0  # for multi_endpoint
        self.safety_signal_active = False   # for safety_monitoring
        self.hidden_toxicity_organ = ""     # organ with hidden signal

        # Trajectory history
        self.trajectory: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # RESET — called at the start of each episode
    # ─────────────────────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> TrialObservation:
        self._reset_rubric()

        if seed is not None:
            random.seed(seed)

        self._init_episode_vars()

        ep_id = episode_id or str(uuid.uuid4())
        task_name = kwargs.get("task_name", "dose_escalation")
        config = TASK_CONFIG.get(task_name, TASK_CONFIG["dose_escalation"])

        difficulty = kwargs.get("difficulty", config.get("difficulty", self.difficulty))
        self.difficulty = difficulty

        # ── Configure hidden truth ──
        low, high = config["optimal_dose_range"]
        self.true_optimal_dose = random.randint(low, high)

        sm_low, sm_high = config["safety_margin_range"]
        self.safe_dose_limit = self.true_optimal_dose + random.randint(sm_low, sm_high)

        # ── Initialize episode ──
        self.week = 1
        self.current_dose = 10
        self.patients_enrolled = 20
        self.starting_budget = float(config["starting_budget"])
        self.budget = self.starting_budget
        self.trial_stopped = False
        self.enrollment_paused = False

        # Initial readings
        self.effectiveness = self._simulate_effectiveness(self.current_dose)
        self.side_effect_rate = self._simulate_side_effects(self.current_dose)
        self.secondary_effectiveness = self._simulate_secondary_endpoint(
            self.current_dose
        )

        # Treatment arms (start with 1 arm at starting dose)
        self.treatment_arms = [
            {"dose": self.current_dose, "patients": 20, "effectiveness": self.effectiveness}
        ]

        # Interim analysis schedule (task-specific)
        if task_name == "interim_analysis":
            self.interim_checkpoints = [8, 16, 24]
        else:
            self.interim_checkpoints = [15, 30]

        # Safety monitoring task — plant hidden toxicity signal
        if task_name == "safety_monitoring":
            self.hidden_toxicity_organ = random.choice(["liver", "kidney", "cardiac"])
        else:
            self.hidden_toxicity_organ = ""

        # Calculate initial statistics
        self._update_statistics()

        self._current_state = TrialState(
            episode_id=ep_id,
            step_count=0,
            task_name=task_name,
            difficulty=difficulty,
            true_optimal_dose=self.true_optimal_dose,
            safe_dose_limit=self.safe_dose_limit,
            doses_visited=[self.current_dose],
        )

        return self._make_observation(
            reward=0.0,
            message=f"Trial started. Task: {task_name}. Initial dose: 10mg on 20 patients.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP — called every time AI makes a decision
    # ─────────────────────────────────────────────────────────────────────────
    def step(
        self,
        action: TrialAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> TrialObservation:
        if self._current_state is None:
            self.reset()

        self._current_state.step_count += 1
        reward = 0.0
        message = ""
        decision = action.decision

        task_name = self._current_state.task_name

        # ── PROCESS ACTION ───────────────────────────────────────────────────

        if decision == "increase_dose":
            old_dose = self.current_dose
            self.current_dose = min(self.current_dose + 20, 150)
            self.budget -= 150_000
            message = f"Increased dose: {old_dose}mg → {self.current_dose}mg."

            self.effectiveness = self._simulate_effectiveness(self.current_dose)
            self.side_effect_rate = self._simulate_side_effects(self.current_dose)
            self.secondary_effectiveness = self._simulate_secondary_endpoint(
                self.current_dose
            )
            self._update_organ_toxicity(self.current_dose)

            # Record dose exploration
            if self.current_dose not in self._current_state.doses_visited:
                self._current_state.doses_visited.append(self.current_dose)

            # Reward based on information gain
            if self.side_effect_rate > 0.30:
                reward = -0.30
                message += " ⚠ HIGH SIDE EFFECTS detected!"
            elif self.effectiveness > 0.40:
                reward = +0.25
                message += " Strong effectiveness observed."
            elif self.effectiveness > 0.25:
                reward = +0.10
                message += " Moderate improvement."
            else:
                reward = -0.05
                message += " No significant improvement."

        elif decision == "decrease_dose":
            old_dose = self.current_dose
            self.current_dose = max(self.current_dose - 15, 5)
            self.budget -= 100_000
            message = f"Decreased dose: {old_dose}mg → {self.current_dose}mg."

            self.effectiveness = self._simulate_effectiveness(self.current_dose)
            self.side_effect_rate = self._simulate_side_effects(self.current_dose)
            self._update_organ_toxicity(self.current_dose)

            if self.current_dose not in self._current_state.doses_visited:
                self._current_state.doses_visited.append(self.current_dose)

            if self.side_effect_rate < 0.15 and self.effectiveness > 0.30:
                reward = +0.15
                message += " Good safety-effectiveness balance."
                self._current_state.safety_responses += 1
            elif self.effectiveness < 0.15:
                reward = -0.10
                message += " Effectiveness dropped too low."
            else:
                reward = +0.05
                message += " Side effects reduced."

        elif decision == "keep_dose":
            self.patients_enrolled += 20
            self.budget -= 80_000
            if self.treatment_arms:
                self.treatment_arms[0]["patients"] += 20
            message = f"Kept dose at {self.current_dose}mg. Enrolled 20 more patients."

            # Penalise excessive data gathering at ineffective doses
            if self.patients_enrolled < 80:
                reward = +0.05
            elif self.effectiveness < 0.20:
                reward = -0.12
                message += " ⚠ Ineffective dose — wasting resources."
            else:
                reward = -0.05
                message += " Sufficient data already collected."

        elif decision == "enroll_more_patients":
            self.patients_enrolled += 40
            self.budget -= 200_000
            if self.treatment_arms:
                self.treatment_arms[0]["patients"] += 40
            message = f"Enrolled 40 patients at {self.current_dose}mg."

            if self.patients_enrolled < 120 and self.effectiveness > 0.20:
                reward = +0.08
            else:
                reward = -0.10
                message += " Unnecessary enrollment."

        elif decision == "stop_trial":
            self.trial_stopped = True
            message = "Trial stopped by agent."

            if self.effectiveness > 0.35 and self.side_effect_rate < 0.20:
                reward = +0.50
                message += " ✓ Effective and safe dose confirmed!"
            elif self.effectiveness < 0.15:
                reward = +0.15
                message += " Correct to stop — drug ineffective."
            elif self.side_effect_rate > 0.30:
                reward = +0.20
                message += " Correct to stop — safety concerns."
            else:
                reward = -0.15
                message += " Premature stop — more data needed."

        elif decision == "add_treatment_arm":
            if len(self.treatment_arms) < 4:
                new_dose = self.current_dose + random.choice([-10, 10, 20])
                new_dose = max(5, min(150, new_dose))
                self.treatment_arms.append(
                    {"dose": new_dose, "patients": 0, "effectiveness": 0.0}
                )
                self.budget -= 250_000
                self._current_state.arms_added += 1
                reward = +0.10 if task_name == "adaptive_enrollment" else +0.05
                message = f"Added treatment arm at {new_dose}mg. Now {len(self.treatment_arms)} arms."
            else:
                reward = -0.05
                message = "Cannot add more arms (max 4)."

        elif decision == "drop_treatment_arm":
            if len(self.treatment_arms) > 1:
                # Drop the worst-performing arm
                worst = min(self.treatment_arms, key=lambda a: a["effectiveness"])
                self.treatment_arms.remove(worst)
                self._current_state.arms_dropped += 1
                reward = +0.10 if task_name == "adaptive_enrollment" else +0.05
                message = f"Dropped arm at {worst['dose']}mg (effectiveness: {worst['effectiveness']:.1%})."
            else:
                reward = -0.05
                message = "Cannot drop the only remaining arm."

        elif decision == "request_interim_analysis":
            self.budget -= 120_000
            self._current_state.interim_analyses_requested += 1
            self.interims_completed += 1

            # Run interim check
            futility = self._calculate_futility()
            if task_name == "interim_analysis":
                reward = +0.15
                message = (
                    f"Interim analysis #{self.interims_completed}: "
                    f"futility_prob={futility:.1%}, power={self.statistical_power:.1%}."
                )
            else:
                reward = +0.05
                message = f"Interim analysis: futility={futility:.1%}."

        elif decision == "pause_enrollment":
            if not self.enrollment_paused:
                self.enrollment_paused = True
                self._current_state.enrollment_paused = True
                self.budget -= 50_000
                if self.side_effect_rate > 0.20 or any(
                    v > 0.25 for v in self.organ_toxicity.values()
                ):
                    reward = +0.20
                    self._current_state.safety_responses += 1
                    message = "Enrollment paused for safety review. ✓ Appropriate response."
                else:
                    reward = -0.10
                    message = "Enrollment paused — but no clear safety signal?"
            else:
                self.enrollment_paused = False
                reward = +0.05
                message = "Enrollment resumed."

        elif decision == "adjust_monitoring":
            self.budget -= 60_000
            # Enhanced monitoring catches more safety signals
            for organ in self.organ_toxicity:
                self.organ_toxicity[organ] = max(
                    0.0, self.organ_toxicity[organ] - 0.03
                )
            if task_name == "safety_monitoring":
                reward = +0.10
                message = "Enhanced safety monitoring activated."
                self._current_state.safety_responses += 1
            else:
                reward = +0.03
                message = "Monitoring protocols adjusted."

        else:
            reward = -0.05
            message = f"Unknown action: {decision}"

        # ── ADVANCE TIME ──────────────────────────────────────────────────────
        self.week += 1
        self.budget -= 50_000  # Weekly running costs

        # Patient dropout (increases with side effects)
        self.dropout_rate = min(0.3, self.side_effect_rate * 0.4 + random.uniform(0, 0.03))
        dropout_count = int(self.patients_enrolled * self.dropout_rate * 0.05)
        self.patients_enrolled = max(10, self.patients_enrolled - dropout_count)

        # Simulate SAEs for safety_monitoring task
        if task_name == "safety_monitoring" and self.current_dose > self.safe_dose_limit * 0.8:
            sae_chance = (self.current_dose - self.safe_dose_limit * 0.8) / 100.0
            if random.random() < sae_chance:
                self.serious_adverse_events += 1
                self._current_state.sae_count += 1
                # Hidden organ signal intensifies
                if self.hidden_toxicity_organ:
                    self.organ_toxicity[self.hidden_toxicity_organ] += random.uniform(0.05, 0.15)

        # Update statistics
        self._update_statistics()

        # Track effectiveness peaks
        self._current_state.max_effectiveness_seen = max(
            self._current_state.max_effectiveness_seen, self.effectiveness
        )
        self._current_state.max_side_effects_seen = max(
            self._current_state.max_side_effects_seen, self.side_effect_rate
        )

        # Update treatment arm effectiveness
        for arm in self.treatment_arms:
            arm["effectiveness"] = self._simulate_effectiveness(arm["dose"])

        # ── CHECK INTERIM ANALYSIS DUE ────────────────────────────────────────
        interim_due = self.week in self.interim_checkpoints

        # ── CHECK IF EPISODE IS DONE ──────────────────────────────────────────
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

        # Clamp reward to [-1.0, +1.0]
        reward = round(max(-1.0, min(1.0, reward)), 4)

        # Record trajectory step
        self.trajectory.append({
            "step": self._current_state.step_count,
            "action": decision,
            "dose": self.current_dose,
            "effectiveness": self.effectiveness,
            "side_effects": self.side_effect_rate,
            "reward": reward,
        })

        return self._make_observation(
            reward=reward,
            message=message,
            done=done,
            interim_due=interim_due,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STATE — metadata about the episode
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def state(self) -> TrialState:
        if self._current_state is None:
            return TrialState()
        return self._current_state

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE — Simulation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _simulate_effectiveness(self, dose: int) -> float:
        """
        Drug effectiveness peaks near true_optimal_dose and drops off.
        Uses a smooth bell curve — agent cannot see this formula.
        """
        distance = abs(dose - self.true_optimal_dose)
        # Bell curve: effectiveness peaks at optimal dose
        effectiveness = math.exp(-(distance ** 2) / (2 * 25 ** 2))
        noise = random.uniform(-0.04, 0.04)
        return round(min(1.0, max(0.0, effectiveness + noise)), 3)

    def _simulate_side_effects(self, dose: int) -> float:
        """
        Side effects increase as dose exceeds safe_dose_limit.
        Sigmoid-shaped: minimal below limit, rapidly increasing above.
        """
        if dose <= self.safe_dose_limit:
            base = dose / (self.safe_dose_limit * 4)
        else:
            excess = dose - self.safe_dose_limit
            base = 0.15 + 0.5 / (1.0 + math.exp(-0.15 * (excess - 15)))

        noise = random.uniform(-0.03, 0.03)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _simulate_secondary_endpoint(self, dose: int) -> float:
        """
        Secondary endpoint effectiveness — peaks at a DIFFERENT dose.
        This creates the multi-objective trade-off.
        """
        secondary_optimal = self.true_optimal_dose - 15  # peaks lower
        distance = abs(dose - secondary_optimal)
        eff = math.exp(-(distance ** 2) / (2 * 30 ** 2))
        noise = random.uniform(-0.05, 0.05)
        return round(min(1.0, max(0.0, eff + noise)), 3)

    def _update_organ_toxicity(self, dose: int):
        """Update per-organ toxicity based on dose vs safety limit."""
        excess_ratio = max(0, (dose - self.safe_dose_limit * 0.7)) / 100.0

        for organ in self.organ_toxicity:
            base_increase = excess_ratio * random.uniform(0.01, 0.06)
            # Hidden organ takes more damage (for safety_monitoring task)
            if organ == self.hidden_toxicity_organ:
                base_increase *= 2.5
            self.organ_toxicity[organ] = round(
                min(1.0, max(0.0, self.organ_toxicity[organ] + base_increase)), 3
            )

    def _update_statistics(self):
        """Recalculate statistical signals based on current data."""
        n = max(1, self.patients_enrolled)

        # Confidence interval narrows with more patients
        se = max(0.01, 0.5 / math.sqrt(n))
        self.confidence_interval_width = 1.96 * se

        # Statistical power increases with sample size and effect size
        effect_size = self.effectiveness
        if effect_size > 0.01:
            ncp = effect_size * math.sqrt(n)
            # Approximate power using normal CDF approximation
            self.statistical_power = round(
                min(0.99, 1.0 - math.exp(-0.5 * ncp)), 3
            )
        else:
            self.statistical_power = 0.05  # baseline

    def _calculate_futility(self) -> float:
        """
        Bayesian predictive probability of trial failure.
        High futility = should probably stop.
        """
        if self.effectiveness < 0.15:
            return round(min(1.0, 0.70 + random.uniform(0, 0.15)), 3)
        elif self.effectiveness < 0.30:
            return round(min(1.0, 0.30 + random.uniform(0, 0.20)), 3)
        else:
            return round(max(0.0, 0.10 + random.uniform(-0.05, 0.10)), 3)

    def _make_observation(
        self,
        reward: float,
        message: str,
        done: bool = False,
        interim_due: bool = False,
    ) -> TrialObservation:
        """Build the full observation with all signals."""
        ci_low = max(0.0, self.effectiveness - self.confidence_interval_width)
        ci_high = min(1.0, self.effectiveness + self.confidence_interval_width)
        futility = self._calculate_futility()
        data_maturity = min(1.0, self.week / 52.0)

        return TrialObservation(
            # Core
            week=self.week,
            current_dose_mg=self.current_dose,
            patients_enrolled=self.patients_enrolled,
            avg_effectiveness=self.effectiveness,
            side_effect_rate=self.side_effect_rate,
            budget_remaining=self.budget,
            message=message,
            # Statistical
            confidence_interval_low=round(ci_low, 3),
            confidence_interval_high=round(ci_high, 3),
            statistical_power=self.statistical_power,
            # Patient dynamics
            dropout_rate=round(self.dropout_rate, 3),
            active_treatment_arms=len(self.treatment_arms),
            patients_per_arm=[a["patients"] for a in self.treatment_arms],
            # Safety
            serious_adverse_events=self.serious_adverse_events,
            organ_toxicity=dict(self.organ_toxicity),
            # Regulatory
            interim_analysis_due=interim_due,
            data_maturity=round(data_maturity, 3),
            futility_probability=futility,
            # Base class
            done=done,
            reward=reward,
        )