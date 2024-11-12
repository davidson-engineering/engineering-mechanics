__version__ = "0.0.1"

from statics.statics import BoundVector, Reaction, Load, ReactionSolver
from statics.statics import (
    UnderconstrainedError,
    IllConditionedError,
    OverconstrainedWarning,
)
