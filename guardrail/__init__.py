__version__ = "0.1.0-alpha"

from .core import (
    snapshot,
    check,
    DriftResult,
    main,          # keeps `python -m guardrail …` working
)

__all__ = ["snapshot", "check", "DriftResult", "main"]