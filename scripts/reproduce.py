"""One-command offline reproduction from the checked-in real-data snapshot."""
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
started = time.perf_counter()
for command in ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                [sys.executable, "-m", "support_lab", "evaluate", "--mode", "diagnostic"],
                [sys.executable, "-m", "support_lab", "status"]):
    subprocess.run(command, cwd=root, check=True)
print(f"Offline reproduction finished in {time.perf_counter()-started:.2f} seconds.")
print("Diagnostic metrics are not human accuracy. Run gold evaluation only after real annotation.")
