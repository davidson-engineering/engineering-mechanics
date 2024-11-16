from abc import abstractmethod
from typing import List
import logging
import warnings
import numpy as np
from scipy.linalg import lstsq, null_space


from base.vector import BoundVector
from base.result import Result

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
    """
    A class to represent a linear solver for systems of linear equations.

    Attributes
    ----------
    _result_factory : Result
        A factory for creating result objects.
    equations : List[np.ndarray]
        A list of numpy arrays representing the equations.
    constants : List[np.ndarray]
        A list of numpy arrays representing the constants.
    preferred_method : callable
        The preferred method for solving the system of equations.

    Methods
    -------
    __init__(equations: List[np.ndarray] = None, constants: List[np.ndarray] = None)
        Initializes the LinearSolver with optional equations and constants.

    construct_terms(equations: List[BoundVector], constants: List[BoundVector]) -> tuple[np.ndarray, np.ndarray]
        Constructs the coefficient matrix and constant vector from the given equations and constants.

    construct_coeff_matrix(equations: List[BoundVector])
        Abstract method to construct the coefficient matrix from the given equations.

    construct_constant_vector(constants: List[BoundVector])
        Abstract method to construct the constant vector from the given constants.

    check_singularity(A: np.ndarray = None)
        Checks if the equilibrium matrix A is singular or nearly singular.

    check_rank(A: np.ndarray = None)
        Checks the rank of the equilibrium matrix A to determine if it is singular or nearly singular.

    check_condition_number(A: np.ndarray = None)
        Checks the condition number of the equilibrium matrix A.

    check_null_space(A: np.ndarray = None)
        Checks if the equilibrium matrix A has a non-trivial null space.

    solve(A: np.ndarray = None, b: np.ndarray = None)
        Solves the system of linear equations using available solvers and returns the solution and a report.
    """

    _result_factory = Result

    def __init__(
        self, equations: List[np.ndarray] = None, constants: List[np.ndarray] = None
    ):
        self.equations = equations
        self.constants = constants
        self.preferred_method = None

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
        A = self.construct_coeff_matrix(self.equations) if A is None else A
        rank = self.check_rank(A)
        condition = self.check_condition_number(A)
        null_space = self.check_null_space(A)

        return rank, condition, null_space

    def check_rank(self, A: np.ndarray = None):
        """Use rank to check if the equilibrium matrix A is singular or nearly singular."""

        A = self.construct_coeff_matrix(self.equations) if A is None else A
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
        A = self.construct_coeff_matrix(self.equations) if A is None else A
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

    def solve(
        self, A: np.ndarray = None, b: np.ndarray = None
    ) -> tuple[np.ndarray, dict]:
        A = self.construct_coeff_matrix() if A is None else A
        b = self.construct_constant_vector(self.constants) if b is None else b
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
        preferred_method = self.preferred_method
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
