from dataclasses import dataclass, field
import numpy as np
from typing import List
from scipy.linalg import lstsq, null_space
import warnings

from mechanics import BoundVector


class UnderconstrainedError(ValueError):
    pass


class IllConditionedError(ValueError):
    pass


class OverconstrainedWarning(Warning):
    pass


def skew_sym(x) -> np.array:
    """
    Return the skew-symmetric matrix of a 3D vector.

    Args:
        x (array-like): A 3-element array representing a vector.

    Returns:
        np.array: 3x3 skew-symmetric matrix.
    """
    if isinstance(x, list):
        x = np.array(x)
    if x.shape == (3,):
        a, b, c = x
    else:
        return None
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])


@dataclass
class Reaction(BoundVector):
    constraint: np.ndarray = field(default_factory=lambda: np.eye(6))

    def __post_init__(self):
        # Check if the constraint is a 1x6 vector
        if self.constraint.shape == (6,):
            # Convert a 1x6 vector into a 6x6 matrix
            self.constraint = np.diag(self.constraint)
        elif self.constraint.shape == (6, 6):
            # Use the 6x6 matrix as-is
            pass
        else:
            raise ValueError("Constraint must be either a 1x6 vector or a 6x6 matrix")


class StaticsCalculator:
    def __init__(
        self,
        forces: List[BoundVector],
        moments: List[BoundVector],
        reactions: List[Reaction],
    ):
        self.forces = forces
        self.moments = moments
        self.reactions = reactions
        self.num_reactions = len(reactions)

    def assemble_equilibrium_matrix(self):
        # A matrix size: 6 x (6 * num_reactions)
        num_eqs = 6
        num_unknowns = 6 * self.num_reactions
        A = np.zeros((num_eqs, num_unknowns))
        b = np.zeros(num_eqs)

        # Step 1: Force equilibrium - sum of external forces
        total_force = np.sum([f.magnitude for f in self.forces], axis=0)
        b[0:3] = -total_force  # Force balance for Fx, Fy, Fz

        # Step 2: Moment equilibrium - sum of moments from forces and applied moments
        total_moment = np.zeros(3)

        # Add moments from external forces at their respective locations
        for f in self.forces:
            total_moment += np.cross(f.location, f.magnitude)

        # Add applied moments (including any additional moments from location offsets)
        for m in self.moments:
            total_moment += m.magnitude  # Direct contribution of applied moment
            total_moment += np.cross(
                m.location, m.magnitude
            )  # Additional moment if offset from reference

        b[3:6] = -total_moment  # Moment balance for Mx, My, Mz

        # Step 3: Populate A matrix with constraints for each reaction
        for i, reaction in enumerate(self.reactions):
            reaction_index = 6 * i  # Column index for this reaction's components
            constraint = reaction.constraint

            # Force constraints for this reaction (first three rows of A)
            A[0:3, reaction_index : reaction_index + 3] = constraint[0:3, 0:3]

            # Calculate the cross-product matrix for the reaction location
            r_cross = skew_sym(reaction.location)

            # Moment contributions from force constraints (moment arm effect)
            A[3:6, reaction_index : reaction_index + 3] = r_cross @ constraint[0:3, 0:3]

            # Direct moment constraints - apply to moment equilibrium rows (last three rows of A)
            A[3:6, reaction_index + 3 : reaction_index + 6] = constraint[3:6, 3:6]

        # Check for singularities
        self.check_singularity(A)
        self.check_null_space(A)

        return A, b

    def check_singularity(self, A):
        """Check if the equilibrium matrix A is singular or nearly singular."""
        rank = np.linalg.matrix_rank(A)
        if rank < min(A.shape):
            raise UnderconstrainedError(
                f"Equilibrium matrix is under-constrained or redundant with rank={rank}. Check constraints."
            )

        cond_number = np.linalg.cond(A)
        if cond_number > 1e12:
            raise IllConditionedError(
                f"Equilibrium matrix is ill-conditioned with condition number={cond_number:.2e}. Check constraints."
            )

    def check_null_space(self, A):
        """Check if the equilibrium matrix A has a non-trivial null space."""
        null_space_ = null_space(A)
        if null_space_.size > 0:
            warnings.warn(
                "System has a non-trivial null space, possibly indicating redundant constraints. Consider removing constraints.\n Continuing with solution."
            )

    def solve_reactions(self):
        A, b = self.assemble_equilibrium_matrix()
        reactions, residuals, rank, s = lstsq(A, b)
        return reactions.reshape((self.num_reactions, 6))
