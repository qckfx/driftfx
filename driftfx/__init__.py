__version__ = "0.1.1"

from .core import (
    snapshot,
    check,
    DriftResult,
    main,          # keeps `python -m driftfx …` working
)

__all__ = ["snapshot", "check", "DriftResult", "main"]