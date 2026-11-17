"""Puts the module root on sys.path so `import toothid` works from any cwd."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
