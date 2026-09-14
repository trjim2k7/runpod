import os
import sys
from pathlib import Path

os.environ.setdefault("ENHANCE_WORKSPACE", str(Path(__file__).parent / "_workspace"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
