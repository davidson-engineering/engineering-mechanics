__version__ = "0.0.1"

from .study import Study
from .solver import (
    LinearSolver,
    UnderconstrainedError,
    OverconstrainedWarning,
    IllConditionedError,
)
