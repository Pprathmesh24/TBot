"""Launch the TBot Streamlit monitoring dashboard."""
import subprocess
import sys
from pathlib import Path

dashboard = Path(__file__).parent.parent / "src" / "tbot" / "monitoring" / "dashboard.py"
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(dashboard), "--server.headless", "false"],
    check=True,
)
