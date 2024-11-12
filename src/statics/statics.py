from dataclasses import dataclass, field
import numpy as np
from typing import List
from scipy.linalg import lstsq, null_space
import warnings
import logging
from prettytable import PrettyTable

from mechanics import BoundVector

logger = logging.getLogger(__name__)


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
class Load(BoundVector):
    magnitude: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def __post_init__(self):
        super().__post_init__()
        if self.magnitude.size < 6:
            self.magnitude = np.pad(self.magnitude, (0, 6 - self.magnitude.size))
        if self.magnitude.size > 6:
            raise ValueError("Magnitude must be a 1x6 vector.")


@dataclass
class Reaction(BoundVector):

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


class StaticsSolver:
    pass


class ReactionSolver(StaticsSolver):
    def __init__(
        self,
        loads: List[BoundVector],
        reactions: List[Reaction],
    ):
        self.loads = loads
        self.reactions = reactions
        self.dof = 6  # Degrees of freedom for each reaction
        self.num_reactions = len(reactions)

        # Pad loads to 6 components if necessary
        for load in self.loads:
            if load.magnitude.size < 6:
                load.magnitude = np.pad(load.magnitude, (0, 6 - load.magnitude.size))

    def assemble_equilibrium_matrix(self):
        # A matrix size: 6 x (6 * num_reactions)
        num_unknowns = self.dof * self.num_reactions
        A = np.zeros((self.dof, num_unknowns))
        b = np.zeros(self.dof)

        # load.magnitude = [Fx, Fy, Fz, Mx, My, Mz]
        # load.location = [x, y, z]

        locations = np.array([l.location for l in self.loads])  # Shape (n_loads, 3)
        magnitudes = np.array([l.magnitude for l in self.loads])  # Shape (n_loads, 6)

        # Step 1: Force equilibrium - sum of external loads
        b = -np.sum(magnitudes, axis=0)

        # Step 2: Moment equilibrium - sum of moments from loads and applied moments
        total_moment = np.zeros(3)

        moments_from_forces = np.cross(
            locations, magnitudes[:, 0:3]
        )  # Shape (n_loads, 3)
        # Sum all moments from forces
        b[3:6] -= np.sum(moments_from_forces, axis=0)

        # Add applied moments
        cross_offsets = np.cross(locations, magnitudes[:, 3:6])  # Shape (n_moments, 3)

        # Sum applied moments and cross products of moments with their locations
        b[3:6] -= np.sum(cross_offsets, axis=0)

        # Step 3: Populate A matrix with constraints for each reaction
        for i, reaction in enumerate(self.reactions):
            reaction_index = self.dof * i  # Column index for this reaction's components
            constraint = reaction.constraint

            # Force constraints for this reaction (first three rows of A)
            A[0:3, reaction_index : reaction_index + 3] = constraint[0:3, 0:3]

            # Calculate the cross-product matrix for the reaction location
            r_cross = skew_sym(reaction.location)

            # Moment contributions from force constraints (moment arm effect)
            A[3:6, reaction_index : reaction_index + 3] = r_cross @ constraint[0:3, 0:3]

            # Direct moment constraints - apply to moment equilibrium rows (last three rows of A)
            A[3:6, reaction_index + 3 : reaction_index + 6] = constraint[3:6, 3:6]

        self.check_singularity(A)

        return A, b

    def check_constraints(self):

        def check_constraint_matrix_size(reaction):
            if reaction.constraint.shape not in [(6,), (6, 6)]:
                raise ValueError(
                    f"Constraint matrix for reaction '{reaction.name}' must be a 1x6 vector or a 6x6 matrix."
                )

        for reaction in self.reactions:
            check_constraint_matrix_size(reaction)

    def check_singularity(self, A):
        self.check_rank(A)
        self.check_condition_number(A)
        self.check_null_space(A)

    def check_rank(self, A):
        """Check if the equilibrium matrix A is singular or nearly singular."""

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

    def check_condition_number(self, A):
        """Check the condition number of the equilibrium matrix A."""
        cond_number = np.linalg.cond(A)
        if cond_number > 1e12:
            raise IllConditionedError(
                f"Equilibrium matrix is ill-conditioned with condition number={cond_number:.2e}. Check constraints."
            )
        return cond_number

    def check_null_space(self, A):
        """Check if the equilibrium matrix A has a non-trivial null space."""
        null_space_ = null_space(A)
        if null_space_.size > 0:
            warnings.warn(
                "System has a non-trivial null space, possibly indicating redundant constraints. Consider removing constraints.\n Continuing with solution."
            )
        return null_space_

    def solve_reactions(self):
        A, b = self.assemble_equilibrium_matrix()

        def generate_report(A, b, result, solver):
            solver_report = {
                "A": A,
                "b": b,
                "rank": np.linalg.matrix_rank(A),
                "condition_number": np.linalg.cond(A),
                "null_space": null_space(A),
                "result": result,
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
                result = solver(A, b).reshape((self.num_reactions, self.dof))
                self.report = generate_report(A, b, result, solver)

                logger.info(f"Solver {solver.__name__} succeeded.")

                return result

            except np.linalg.LinAlgError:
                logger.warning(f"Solver {solver} failed. Trying next solver.")

        logger.error("All solvers failed. No solution found.")
        return None

    def print_summary(self, reactions_result, decimal_places=2):
        """
        Print summaries of input loads, constraints, and reactions using PrettyTable.

        Args:
            reactions_result (list): List of reaction results from solve_reactions(),
                                     each entry should contain [Fx, Fy, Fz, Mx, My, Mz].
            decimal_places (int): Number of decimal places to display for numeric values.
        """
        float_format = f"{{:.{decimal_places}f}}"

        # Input Loads Table
        loads_table = PrettyTable()
        loads_table.field_names = [
            "Load",
            "Loc X",
            "Loc Y",
            "Loc Z",
            "Fx",
            "Fy",
            "Fz",
            "Mx",
            "My",
            "Mz",
        ]

        for i, load in enumerate(self.loads):
            name = load.name if load.name else f"Force {i+1}"
            loc_x, loc_y, loc_z = load.location
            fx, fy, fz, mx, my, mz = load.magnitude
            loads_table.add_row(
                [
                    name,
                    float_format.format(loc_x),
                    float_format.format(loc_y),
                    float_format.format(loc_z),
                    float_format.format(fx),
                    float_format.format(fy),
                    float_format.format(fz),
                    float_format.format(mx),
                    float_format.format(my),
                    float_format.format(mz),
                ]
            )
        # Constraints Table
        constraints_table = PrettyTable()
        constraints_table.field_names = [
            "Reaction",
            "Constraint Matrix",
        ]

        for i, reaction in enumerate(self.reactions):
            loc_x, loc_y, loc_z = reaction.location
            # Convert constraint matrix to multi-line string
            constraint_matrix = reaction.constraint
            if isinstance(constraint_matrix, (list, np.ndarray)):
                constraint_matrix_str = "\n".join(
                    [
                        "[" + " ".join((f"{value:g}") for value in row) + "]"
                        for row in constraint_matrix
                    ]
                )
            else:
                constraint_matrix_str = str(
                    constraint_matrix
                )  # Use string representation for non-matrix types

            constraints_table.add_row(
                [
                    reaction.name,
                    constraint_matrix_str,
                ]
            )

        # Results Table
        reactions_table = PrettyTable()
        reactions_table.field_names = [
            "Reaction",
            "Loc X",
            "Loc Y",
            "Loc Z",
            "Fx",
            "Fy",
            "Fz",
            "Mx",
            "My",
            "Mz",
        ]

        for i, reaction in enumerate(self.reactions):
            loc_x, loc_y, loc_z = reaction.location
            # Concatenate loads and moments
            reaction_values = reactions_result[i]
            row = [
                reaction.name,
                float_format.format(loc_x),
                float_format.format(loc_y),
                float_format.format(loc_z),
                *[float_format.format(val) for val in reaction_values],
            ]
            reactions_table.add_row(row)

        # Set alignment: left-align the first column, center-align others
        loads_table.align["Load"] = "l"
        constraints_table.align["Reaction"] = "l"
        reactions_table.align["Reaction"] = "l"

        for col in loads_table.field_names[1:]:
            loads_table.align[col] = "c"
        for col in constraints_table.field_names[1:]:
            constraints_table.align[col] = "c"
        for col in reactions_table.field_names[1:]:
            reactions_table.align[col] = "c"

        # ANSI color codes
        green = "\033[32m"  # Green color
        grey = "\033[90m"  # Grey color
        reset_color = "\033[0m"  # Reset color

        def print_table_with_colored_borders(
            table, vertical_border_color=green, horizontal_border_color=grey
        ):
            """Helper function to print table with green horizontal borders only."""
            table_str = table.get_string()
            for line in table_str.splitlines():
                # Color only the horizontal lines (contains only +, -)
                if set(line.strip()) <= {"+", "-", "="}:
                    print(f"{vertical_border_color}{line}{reset_color}")
                else:
                    # Apply grey color to vertical borders and reset color for text
                    colored_line = line.replace(
                        "|", f"{horizontal_border_color}|{reset_color}"
                    )
                    print(colored_line)

        # Print all tables
        print("Input loads Summary:")
        print_table_with_colored_borders(loads_table)
        print("\nConstraints Summary:")
        print_table_with_colored_borders(constraints_table)
        print("\nReactions Summary:")
        print_table_with_colored_borders(reactions_table)

        # print(self.report)

    def run(self):
        reactions_result = self.solve_reactions()
        self.print_summary(reactions_result)
