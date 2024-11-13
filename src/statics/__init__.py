__version__ = "0.1.0"

from statics.statics import BoundVector, Reaction, Load, ReactionSolver
from statics.statics import (
    UnderconstrainedError,
    IllConditionedError,
    OverconstrainedWarning,
)
