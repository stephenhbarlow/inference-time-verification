import os
from pathlib import Path
import subprocess
import sys


def test_complete_synthetic_example():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/synthetic/run_example.py"],
        cwd=root,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root / "src"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Synthetic ITV example passed." in result.stdout
