"""Always test the standalone public source, never the parent repository."""

import sys
from pathlib import Path

# Some Windows wheels share MSVC DLLs; load the encoder runtime first.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
