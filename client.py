"""
Clinical Trial Optimization — OpenEnv Client
Programmatic client for interacting with the Clinical Trial environment
via the OpenEnv HTTP/WebSocket API.

Usage:
    from client import ClinicalTrialEnvClient, TrialAction

    client = ClinicalTrialEnvClient(base_url="http://localhost:7860")
    obs = client.reset(task_name="dose_escalation")
    obs = client.step(TrialAction(decision="increase_dose"))
    print(obs.avg_effectiveness, obs.side_effect_rate)
    client.close()
"""
import requests
from typing import Optional
from models import TrialAction, TrialObservation


class ClinicalTrialEnvClient:
    """HTTP client for the Clinical Trial OpenEnv server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> dict:
        """Check server health."""
        r = self.session.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def tasks(self) -> list:
        """List all available tasks."""
        r = self.session.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()["tasks"]

    def reset(self, task_name: str = "dose_escalation", **kwargs) -> dict:
        """Reset the environment for a specific task."""
        r = self.session.post(f"{self.base_url}/reset/{task_name}")
        r.raise_for_status()
        return r.json()

    def step(self, action: TrialAction) -> dict:
        """Take one step in the environment."""
        r = self.session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        """Get current episode state/metadata."""
        r = self.session.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def grade(self) -> dict:
        """Grade the current episode."""
        r = self.session.post(f"{self.base_url}/grade")
        r.raise_for_status()
        return r.json()

    def schema(self) -> dict:
        """Get action/observation JSON schemas."""
        r = self.session.get(f"{self.base_url}/schema")
        r.raise_for_status()
        return r.json()

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Quick demo
    print("Clinical Trial Env Client — Demo")
    print("=" * 40)

    client = ClinicalTrialEnvClient()

    try:
        health = client.health()
        print(f"Server health: {health}")
    except Exception as e:
        print(f"Server not running at localhost:7860: {e}")
        print("Start the server first: uvicorn server.app:app --port 7860")
        exit(1)

    tasks = client.tasks()
    print(f"Available tasks: {[t['name'] for t in tasks]}")

    result = client.reset("dose_escalation")
    print(f"Reset: dose={result['observation']['current_dose_mg']}mg")

    result = client.step(TrialAction(decision="increase_dose"))
    print(f"Step: dose={result['observation']['current_dose_mg']}mg, "
          f"effectiveness={result['observation']['avg_effectiveness']}")

    grade = client.grade()
    print(f"Grade: {grade['score']}")

    client.close()
    print("Done.")
