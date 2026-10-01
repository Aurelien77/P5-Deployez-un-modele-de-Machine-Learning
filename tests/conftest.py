"""Rend `import backend` possible sans PYTHONPATH, sous Windows comme en CI."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "backend"
root = str(_ROOT)
backend = str(_BACKEND)
if root not in sys.path:
    sys.path.insert(0, root)
# Append, jamais insert(0) : sinon le dossier backend masque le paquet `backend`.
if backend not in sys.path:
    sys.path.append(backend)
