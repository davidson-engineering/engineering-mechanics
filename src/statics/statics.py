from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import List

import numpy as np
from prettytable import PrettyTable
from simulation.result import Result
from simulation.solver import LinearSolver

logger = logging.getLogger(__name__)

from mechanics import Load


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
class Reaction(Load):

    constraint: np.ndarray = field(default_factory=lambda: np.eye(6))

    def __post_init__(self):
        super().__post_init__()
        # Check if the constraint is a 1x6 vector
        if self.constraint.shape == (6,):
            # Convert a 1x6 vector into a 6x6 matrix
            self.constraint = np.diag(self.constraint)
        elif self.constraint.shape == (6, 6):
            # Use the 6x6 matrix as-is
            pass
        else:
            raise ValueError("Constraint must be either a 1x6 vector or a 6x6 matrix")


class ReactionSolver(LinearSolver):

    def __init__(
        self,
        loads: List[Load],
        reactions: List[Reaction],
    ):
        super().__init__(equations=reactions, constants=loads)

        self.check_constraints()

    @property
    def loads(self):
        return self.constants

    @property
    def reactions(self):
        return self.equations

    def construct_constant_vector(self, loads: List[Load] = None) -> np.ndarray:
        """Construct the right-hand side vector b for the equilibrium equations."""
        loads = loads or self.loads
        b = np.zeros(6)

        locations = np.array([load.location for load in loads])  # Shape (n_loads, 3)
        magnitudes = np.array([load.magnitude for load in loads])  # Shape (n_loads, 6)

        # Step 1: Force equilibrium - sum of external loads
        b = -np.sum(magnitudes, axis=0)

        moments_from_forces = np.cross(
            locations, magnitudes[:, 0:3]
        )  # Shape (n_loads, 3)
        # Sum all moments from forces
        b[3:6] -= np.sum(moments_from_forces, axis=0)

        # Add applied moments
        cross_offsets = np.cross(locations, magnitudes[:, 3:6])  # Shape (n_moments, 3)

        # Sum applied moments and cross products of moments with their locations
        b[3:6] -= np.sum(cross_offsets, axis=0)

        return b

    def construct_coeff_matrix(self, reactions: List[Reaction] = None) -> np.ndarray:
        """Construct the equilibrium matrix A for the reaction forces and moments."""

        reactions = reactions or self.reactions

        A = np.zeros((6, 6 * len(reactions)))

        for i, reaction in enumerate(reactions):
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

        return A

    def check_constraints(self):

        def check_constraint_matrix_size(reaction):
            if reaction.constraint.shape not in [(6,), (6, 6)]:
                raise ValueError(
                    f"Constraint matrix for reaction '{reaction.name}' must be a 1x6 vector or a 6x6 matrix."
                )

        for reaction in self.reactions:
            check_constraint_matrix_size(reaction)
