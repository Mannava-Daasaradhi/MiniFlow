import math
import subprocess
import uuid
from datetime import datetime, timezone


def generate_run_id() -> str:
    """Returns str(uuid.uuid4())."""
    return str(uuid.uuid4())


def get_git_hash() -> str | None:
    """
    Runs: git rev-parse --short HEAD.
    Returns 7-char hash string, or None if not in a git repo or git is missing.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def now_iso() -> str:
    """Returns datetime.now(timezone.utc).isoformat()."""
    return datetime.now(timezone.utc).isoformat()


def validate_finite(value: float, name: str) -> None:
    """Raises ValueError if value is NaN or infinite."""
    if not math.isfinite(value):
        raise ValueError(f"Metric '{name}' must be a finite number, got: {value}")
