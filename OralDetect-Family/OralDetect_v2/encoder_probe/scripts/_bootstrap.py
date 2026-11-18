"""Puts the module root on sys.path so `import encprobe` works from any cwd."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
