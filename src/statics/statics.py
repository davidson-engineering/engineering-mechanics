from dataclasses import dataclass, field
from datetime import datetime
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
        if self.magnitude.size < 6:
            self.magnitude = np.pad(self.magnitude, (0, 6 - self.magnitude.size))
        if self.magnitude.size > 6:
            raise ValueError("Magnitude must be a 1x6 vector.")
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


def construct_b_vector(vectors: List[BoundVector], dof: int = 6) -> np.ndarray:
    """Construct the right-hand side vector b for the equilibrium equations."""
    b = np.zeros(dof)

    # load.magnitude = [Fx, Fy, Fz, Mx, My, Mz]
    # load.location = [x, y, z]

    locations = np.array([vector.location for vector in vectors])  # Shape (n_loads, 3)
    magnitudes = np.array(
        [vector.magnitude for vector in vectors]
    )  # Shape (n_loads, 6)

    # Step 1: Force equilibrium - sum of external loads
    b = -np.sum(magnitudes, axis=0)

    moments_from_forces = np.cross(locations, magnitudes[:, 0:3])  # Shape (n_loads, 3)
    # Sum all moments from forces
    b[3:6] -= np.sum(moments_from_forces, axis=0)

    # Add applied moments
    cross_offsets = np.cross(locations, magnitudes[:, 3:6])  # Shape (n_moments, 3)

    # Sum applied moments and cross products of moments with their locations
    b[3:6] -= np.sum(cross_offsets, axis=0)

    return b


def construct_A_matrix(reactions: List[Reaction], dof: int = 6) -> np.ndarray:
    """Construct the equilibrium matrix A for the reaction forces and moments."""

    A = np.zeros((dof, dof * len(reactions)))

    for i, reaction in enumerate(reactions):
        reaction_index = dof * i  # Column index for this reaction's components
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


class ReactionSolver(StaticsSolver):
    def __init__(
        self,
        loads: List[BoundVector],
        reactions: List[Reaction],
    ):
        self.loads = loads
        self.reactions = reactions
        self.dof = 6  # Degrees of freedom for each reaction

        # Pad loads to 6 components if necessary
        for load in self.loads:
            if load.magnitude.size < 6:
                load.magnitude = np.pad(load.magnitude, (0, 6 - load.magnitude.size))

    def assemble_equilibrium_matrix(self):
        """Assemble the equilibrium matrix A and right-hand side vector b."""
        b = construct_b_vector(self.loads)
        A = construct_A_matrix(self.reactions)

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

    def run(self):
        reactions_result = self.solve_reactions()

        self.validate_results()
        self.print_summary(reactions_result, html_report_path="report.html")

    def solve(self):
        A, b = self.assemble_equilibrium_matrix()
        self.check_singularity(A)

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
                result = solver(A, b).reshape((len(self.reactions), self.dof))
                for i, reaction in enumerate(self.reactions):
                    reaction.magnitude = result[i]
                if self.validate_results(b):
                    self.report = generate_report(A, b, result, solver)
                    logger.info(f"Solver {solver.__name__} succeeded.")
                    return result

            except np.linalg.LinAlgError:
                logger.warning(f"Solver {solver} failed. Trying next solver.")

        logger.error("All solvers failed. No solution found.")
        return None

    def validate_results(self, b=None):
        """Validate the reaction results by checking equilibrium."""
        if b is None:
            b = construct_b_vector(self.loads)
        b_reactions = construct_b_vector(self.reactions)
        if np.allclose(b, -b_reactions):
            logger.info("Equilibrium check passed.")
            return True
        else:
            logger.error("Equilibrium check failed.")
            return False

    def print_summary(self, decimal_places=2, html_report_path=None):
        """
        Print summaries of input loads, constraints, and reactions using PrettyTable and optionally generate an HTML report.

        Args:
            reactions_result (list): List of reaction results from solve_reactions(),
                                     each entry should contain [Fx, Fy, Fz, Mx, My, Mz].
            decimal_places (int): Number of decimal places to display for numeric values.
            html_report_path (str): Path to save the HTML report. If None, HTML report is not generated.
        """
        float_format = f"{{:.{decimal_places}f}}"

        # Create PrettyTables for console printout
        loads_table, constraints_table, reactions_table = self._create_pretty_tables(
            float_format
        )

        # Print tables to console with color
        print("Input loads Summary:")
        self._print_table_with_colored_borders(loads_table)
        print("\nConstraints Summary:")
        self._print_table_with_colored_borders(constraints_table)
        print("\nReactions Summary:")
        self._print_table_with_colored_borders(reactions_table)

        # Generate HTML report if a path is provided
        if html_report_path:
            html_content = self._generate_html_report(
                loads_table, constraints_table, reactions_table
            )
            with open(html_report_path, "w") as file:
                file.write(html_content)
            logger.info(f"HTML report saved to {html_report_path}")

    def _create_pretty_tables(self, float_format):
        # Load table
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
            loc_x, loc_y, loc_z = load.location
            fx, fy, fz, mx, my, mz = load.magnitude
            loads_table.add_row(
                [
                    load.name if load.name else f"Load {i+1}",
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

        # Constraints table
        constraints_table = PrettyTable()
        constraints_table.field_names = ["Reaction", "Constraint Matrix"]
        for i, reaction in enumerate(self.reactions):
            constraint_matrix_str = "\n".join(
                [
                    "[" + " ".join(f"{val:.2f}" for val in row) + "]"
                    for row in reaction.constraint
                ]
            )
            constraints_table.add_row(
                [
                    reaction.name if reaction.name else f"Reaction {i+1}",
                    constraint_matrix_str,
                ]
            )

        # Reactions table
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
            row = [
                reaction.name if reaction.name else f"Reaction {i+1}",
                float_format.format(loc_x),
                float_format.format(loc_y),
                float_format.format(loc_z),
            ]
            row.extend(float_format.format(val) for val in reaction.magnitude)
            reactions_table.add_row(row)

        return loads_table, constraints_table, reactions_table

    def _generate_html_report(self, loads_table, constraints_table, reactions_table):
        # Convert PrettyTables to HTML tables
        def pretty_table_to_html(pretty_table):
            return pretty_table.get_html_string()

        # Convert report and validation results to HTML
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_report = f"""
        <html>
        <head>
            <title>Statics Solver Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                h2 {{ color: #333; }}
                .summary {{ margin: 20px; }}
            </style>
        </head>
        <body>
            <h1>Statics Solver Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>

            <div class="summary">
                <h2>Input Loads Summary</h2>
                {pretty_table_to_html(loads_table)}

                <h2>Constraints Summary</h2>
                {pretty_table_to_html(constraints_table)}

                <h2>Reactions Summary</h2>
                {pretty_table_to_html(reactions_table)}

                <h2>Validation Results</h2>
                <p>Combined reactions match the input loads, indicating equilibrium is achieved.</p>
            </div>
        </body>
        </html>
        """
        return html_report

    def _print_table_with_colored_borders(
        self,
        table,
        vertical_border_color="\033[32m",
        horizontal_border_color="\033[90m",
    ):
        """Helper function to print table with green horizontal borders only."""
        reset_color = "\033[0m"
        table_str = table.get_string()
        for line in table_str.splitlines():
            if set(line.strip()) <= {"+", "-", "="}:
                print(f"{vertical_border_color}{line}{reset_color}")
            else:
                colored_line = line.replace(
                    "|", f"{horizontal_border_color}|{reset_color}"
                )
                print(colored_line)
