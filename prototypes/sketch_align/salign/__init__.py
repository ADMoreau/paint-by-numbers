"""Aligning a target image to a hand-drawn sketch on the canvas (feasibility prototype)."""

import sys
from pathlib import Path

# the X-tracking prototype's camera model and rendering effects are reused
_XT = Path(__file__).resolve().parents[2] / "x_tracking"
if str(_XT) not in sys.path:
    sys.path.insert(0, str(_XT))
