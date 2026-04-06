"""Quick verification script for the Clinical Trial Env."""
import sys, random
sys.path.insert(0, '.')

from server.environment import ClinicalTrialEnvironment
from server.grader import grade_by_task
from models import TrialAction
from openenv.core.env_server import Action, Observation, State, Environment

# ── Test 1: inheritance ──
assert issubclass(ClinicalTrialEnvironment, Environment), "Must inherit from Environment"
print("PASS: ClinicalTrialEnvironment inherits from openenv Environment")

# ── Test 2: easy difficulty ──
random.seed(42)
env_e = ClinicalTrialEnvironment(difficulty='easy')
obs = env_e.reset(task_name='dose_finding_easy')
assert 30 <= env_e.true_optimal_dose <= 70, f"Easy dose out of range: {env_e.true_optimal_dose}"
assert obs.week == 1
assert obs.current_dose_mg == 10
print(f"PASS: Easy episode - optimal_dose={env_e.true_optimal_dose}mg, safety_limit={env_e.safe_dose_limit}mg")

# ── Test 3: medium difficulty ──
random.seed(42)
env_m = ClinicalTrialEnvironment(difficulty='medium')
env_m.reset(task_name='dose_finding_medium')
assert 50 <= env_m.true_optimal_dose <= 80, f"Medium dose out of range: {env_m.true_optimal_dose}"
print(f"PASS: Medium episode - optimal_dose={env_m.true_optimal_dose}mg, safety_limit={env_m.safe_dose_limit}mg")

# ── Test 4: hard difficulty ──
random.seed(42)
env_h = ClinicalTrialEnvironment(difficulty='hard')
env_h.reset(task_name='dose_finding_hard')
assert 60 <= env_h.true_optimal_dose <= 100, f"Hard dose out of range: {env_h.true_optimal_dose}"
print(f"PASS: Hard episode - optimal_dose={env_h.true_optimal_dose}mg, safety_limit={env_h.safe_dose_limit}mg")

# ── Test 5: step works and state increments ──
obs2 = env_e.step(TrialAction(decision='increase_dose'))
assert env_e.state.step_count == 1
assert obs2.reward is not None
assert -1.0 <= float(obs2.reward) <= 1.0, f"Reward out of range: {obs2.reward}"
print(f"PASS: step() - reward={obs2.reward}, step_count={env_e.state.step_count}")

# ── Test 6: grader returns 0.0-1.0 ──
score_e, reasons_e = grade_by_task(env_e, 'dose_finding_easy')
assert 0.0 <= score_e <= 1.0, f"Score out of range: {score_e}"
print(f"PASS: Grader score={score_e:.3f} (in [0.0, 1.0])")

# ── Test 7: task_name stored in state ──
assert env_e.state.task_name == 'dose_finding_easy', f"task_name={env_e.state.task_name}"
assert env_e.state.difficulty == 'easy', f"difficulty={env_e.state.difficulty}"
print("PASS: State stores task_name and difficulty correctly")

print("\nALL TESTS PASSED")
