from dataclasses import dataclass, field
from abc import abstractmethod
from datetime import datetime
import numpy as np
from typing import List
from scipy.linalg import lstsq, null_space
import warnings
import logging

from mechanics.mechanics import Load


logger = logging.getLogger(__name__)

from mechanics import BoundVector
from statics.result import ReactionResult


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


class LinearSolver:

    _result_factory = ReactionResult

    def __init__(self, equations: List[BoundVector], constants: List[BoundVector]):
        self.equations = equations
        self.constants = constants

    def construct_terms(
        self, equations: List[BoundVector], constants: List[BoundVector]
    ) -> tuple[np.ndarray, np.ndarray]:
        A = self.construct_coeff_matrix(equations)
        b = self.construct_constant_vector(constants)
        return A, b

    @abstractmethod
    def construct_coeff_matrix(self, equations: List[BoundVector]): ...

    @abstractmethod
    def construct_constant_vector(self, constants: List[BoundVector]): ...

    def check_singularity(self, A: np.ndarray = None):
        """Check if the equilibrium matrix A is singular or nearly singular."""
        A = self.construct_coeff_matrix() if A is None else A
        rank = self.check_rank(A)
        condition = self.check_condition_number(A)
        null_space = self.check_null_space(A)

        return rank, condition, null_space

    def check_rank(self, A: np.ndarray = None):
        """Use rank to check if the equilibrium matrix A is singular or nearly singular."""

        A = self.construct_coeff_matrix() if A is None else A
        rank = np.linalg.matrix_rank(A)
        if rank < min(A.shape):
            raise UnderconstrainedError(
                f"Equilibrium matrix is under-constrained or redundant with rank={rank}. Check constraints."
            )
        elif rank > min(A.shape):
            warnings.warn(
                f"Equilibrium matrix is over-constrained with rank={rank}. Check constraints.",
                OverconstrainedWarning,
            )
        return rank

    def check_condition_number(self, A: np.ndarray = None):
        """Check the condition number of the equilibrium matrix A."""
        A = self.construct_coeff_matrix() if A is None else A
        cond_number = np.linalg.cond(A)
        if cond_number > 1e12:
            raise IllConditionedError(
                f"Equilibrium matrix is ill-conditioned with condition number={cond_number:.2e}. Check constraints."
            )
        return cond_number

    def check_null_space(self, A: np.ndarray = None):
        """Check if the equilibrium matrix A has a non-trivial null space."""
        A = self.construct_coeff_matrix() if A is None else A
        null_space_ = null_space(A)
        if null_space_.size > 0:
            warnings.warn(
                "System has a non-trivial null space, possibly indicating redundant constraints. Consider removing constraints.\n Continuing with solution."
            )
        return null_space_

    def validate_result(self, result):
        """Validate the reaction results by checking equilibrium."""
        assert isinstance(result, self._result_factory)
        b = self.construct_constant_vector(result.constants)
        b_equations = self.construct_constant_vector(result.equations)
        if np.allclose(b, -b_equations):
            logger.info("Equilibrium check passed.")
            return True
        else:
            logger.error("Equilibrium check failed.")
            return False

    def run(self):
        solution, report = self.solve()
        result = self.build_result(solution, report)
        self.validate_result(result)
        return result
        # self.print_summary(reactions_result, html_report_path="report.html")

    def build_result(self, solution, report=None):
        return self._result_factory(self.equations, self.constants, solution, report)

    def solve(self):
        A = self.construct_coeff_matrix()
        b = self.construct_constant_vector()
        rank, condition, null_space = self.check_singularity(A)

        def generate_report(solver):
            solver_report = {
                "coefficient_matrix": A,
                "constant_vector": b,
                "rank": rank,
                "condition_number": condition,
                "null_space": null_space,
                "solver": solver.__name__,
            }
            return solver_report

        def direct_solver(A, b):
            return np.linalg.solve(A, b)

        def leastsquares_solver(A, b):
            return lstsq(A, b)[0]

        solvers = [direct_solver, leastsquares_solver]
        for solver in solvers:
            try:
                solution = solver(A, b).reshape((len(self.equations), len(b)))
                report = generate_report(solver)
                logger.info(
                    f"Solution found using {solver.__name__}.", extra={"report": report}
                )
                return solution, report

            except np.linalg.LinAlgError:
                logger.warning(f"Solver {solver} failed. Trying next solver.")

        logger.error("All solvers failed. No solution found.")
        return None


class ReactionSolver(LinearSolver):

    _solution_factory = ReactionResult

    def __init__(
        self,
        loads: List[BoundVector],
        reactions: List[Reaction],
    ):
        self.loads = loads
        self.reactions = reactions

        self.check_constraints()

    @property
    def constants(self):
        return self.loads

    @property
    def equations(self):
        return self.reactions

    def construct_constant_vector(self, loads: List[BoundVector] = None) -> np.ndarray:
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
