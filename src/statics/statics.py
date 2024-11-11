from dataclasses import dataclass, field
import numpy as np
from typing import List
from scipy.linalg import lstsq, null_space
import warnings

from prettytable import PrettyTable

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
        loads_table.field_names = ["Load", "Loc X", "Loc Y", "Loc Z", "Fx", "Fy", "Fz"]

        for i, load in enumerate(self.forces):
            loc_x, loc_y, loc_z = load.location
            fx, fy, fz = load.magnitude
            loads_table.add_row(
                [
                    f"Load {i + 1}",
                    float_format.format(loc_x),
                    float_format.format(loc_y),
                    float_format.format(loc_z),
                    float_format.format(fx),
                    float_format.format(fy),
                    float_format.format(fz),
                ]
            )

        # Constraints Table
        constraints_table = PrettyTable()
        constraints_table.field_names = [
            "Reaction",
            "Constraint Matrix",
            "Loc X",
            "Loc Y",
            "Loc Z",
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
                    float_format.format(loc_x),
                    float_format.format(loc_y),
                    float_format.format(loc_z),
                ]
            )

        # Results Table
        reactions_table = PrettyTable()
        reactions_table.field_names = [
            "Reaction",
            "Fx",
            "Fy",
            "Fz",
            "Mx",
            "My",
            "Mz",
        ]

        for i, reaction in enumerate(self.reactions):
            loc_x, loc_y, loc_z = reaction.location
            # Concatenate forces and moments
            reaction_values = reactions_result[i]
            row = [
                reaction.name,
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

        def print_table_with_green_borders(table):
            """Helper function to print table with green horizontal borders only."""
            table_str = table.get_string()
            for line in table_str.splitlines():
                # Color only the horizontal lines (contains only +, -)
                if set(line.strip()) <= {"+", "-", "="}:
                    print(f"{green}{line}{reset_color}")
                else:
                    # Apply grey color to vertical borders and reset color for text
                    colored_line = line.replace("|", f"{grey}|{reset_color}")
                    print(colored_line)

        # Print all tables
        print("Input Loads Summary:")
        print_table_with_green_borders(loads_table)
        print("\nConstraints Summary:")
        print_table_with_green_borders(constraints_table)
        print("\nReactions Summary:")
        print_table_with_green_borders(reactions_table)

    def run(self):
        reactions_result = self.solve_reactions()
        self.print_summary(reactions_result)
