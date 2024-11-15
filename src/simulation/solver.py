from abc import abstractmethod
from typing import List
import logging
import warnings
import numpy as np
from scipy.linalg import lstsq, null_space


from common.types import BoundVector
from simulation.result import Result

logger = logging.getLogger(__name__)


class UnderconstrainedError(ValueError):
    pass


class IllConditionedError(ValueError):
    pass


class OverconstrainedWarning(Warning):
    pass


def direct_solver(A, b):
    return np.linalg.solve(A, b)


def leastsquares_solver(A, b):
    return lstsq(A, b)[0]


class LinearSolver:

    _result_factory = Result

    def __init__(
        self, equations: List[np.ndarray] = None, constants: List[np.ndarray] = None
    ):
        self.equations = equations
        self.constants = constants
        self.preffered_method = None

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

        # self.print_summary(reactions_result, html_report_path="report.html")

    def solve(self, A: np.ndarray = None, b: np.ndarray = None):
        A = self.construct_coeff_matrix() if A is None else A
        b = self.construct_constant_vector() if b is None else b
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

        solvers = [direct_solver, leastsquares_solver]
        # Reorder solvers if a valid preferred method is set
        preferred_method = self.preffered_method
        if preferred_method is not None:
            if preferred_method in solvers:
                solvers = [preferred_method] + [
                    s for s in solvers if s != preferred_method
                ]
            else:
                logger.warning(
                    f"Preferred method {preferred_method} is not a valid solver. Ignoring."
                )

        for solver in solvers:
            try:
                sol_shape = (int(A.shape[1] / len(b)), len(b))
                solution = solver(A, b).reshape(sol_shape)
                report = generate_report(solver)
                logger.debug(
                    f"Solution found using {solver.__name__}.", extra={"report": report}
                )
                return solution, report

            except np.linalg.LinAlgError:
                logger.warning(f"Solver {solver} failed. Trying next solver.")

        logger.error("All solvers failed. No solution found.")
        return None
