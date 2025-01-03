from dataclasses import dataclass
from typing import Any, List

from prettytable import PrettyTable

from base.assembly import Assembly
from base.result import Result
from base.vector import Load, Reaction
from statics.solver import logger


@dataclass
class TableOptions:
    headers: List[str]
    data: List[List[Any]]
    float_format: str = "{:.2f}"
    # vertical_border_color: str = "\033[32m"
    # horizontal_border_color: str = "\033[90m"


class ReactionResult(Result):
    def __init__(self, reactions, loads, solution, report=None):
        super().__init__(equations=reactions, constants=loads)
        self.loads = loads
        self.report = report
        self.reactions = self.update_equations(reactions, solution)

    def update_equations(self, reactions, solution):
        for i, reaction in enumerate(reactions):
            reaction.magnitude = solution[i]
        return reactions

    @property
    def loads(self):
        return self.constants

    @loads.setter
    def loads(self, value):
        self.constants = value

    @property
    def reactions(self):
        return self.equations

    @reactions.setter
    def reactions(self, value):
        self.equations = value

    def print_summary(self, decimal_places=2, html_report_path=None):
        """
        Print summaries of input loads, constraints, and reactions using PrettyTable and optionally generate an HTML report.

        Args:
            reactions_result (list): List of reaction results from solve_reactions(),
                                     each entry should contain [Fx, Fy, Fz, Mx, My, Mz].
            decimal_places (int): Number of decimal places to display for numeric values.
            html_report_path (str): Path to save the HTML report. If None, HTML report is not generated.
        """
        float_format = f"{{:.{decimal_places}g}}"

        # Create PrettyTables for console printout
        loads_table = self.construct_load_table()
        constraints_table = self.construct_constraints_table()
        reactions_table = self.construct_reactions_table()

        # Print tables to console with color
        print("Input loads Summary:")
        print(loads_table)
        print("\nConstraints Summary:")
        print(constraints_table)
        print("\nReactions Summary:")
        print(reactions_table)

        # Generate HTML report if a path is provided
        if html_report_path:
            html_content = self._generate_html_report(
                loads_table, constraints_table, reactions_table
            )
            with open(html_report_path, "w") as file:
                file.write(html_content)
            logger.info(f"HTML report saved to {html_report_path}")

    def _create_pretty_table(self, headers, data, float_format):
        table = PrettyTable()
        table.field_names = headers
        for row in data:
            row_fmt = []
            for val in row:
                try:
                    val = float_format.format(val)
                except ValueError:
                    pass
                row_fmt.append(val)
            table.add_row(row_fmt)
        return table

    def construct_load_table(self):
        load_table_options = TableOptions(
            headers=[
                "Load",
                "X",
                "Y",
                "Z",
                "Fx",
                "Fy",
                "Fz",
                "Mx",
                "My",
                "Mz",
            ],
            data=[
                [
                    load.name if load.name else f"Load {i+1}",
                    *load.location,
                    *load.magnitude,
                ]
                for i, load in enumerate(self.loads)
            ],
            float_format="{:.2f}",
        )
        return self._create_pretty_table(**vars(load_table_options))

    def construct_constraints_table(self):
        float_format = f"{{:2g}}"

        constraints_table_options = TableOptions(
            headers=["Reaction", "Constraint Matrix"],
            data=[
                [
                    reaction.name if reaction.name else f"Reaction {i+1}",
                    "\n".join(
                        [
                            "["
                            + " ".join(float_format.format(val) for val in row)
                            + "]"
                            for row in reaction.constraint
                        ]
                        + ["\n"]
                    ),
                ]
                for i, reaction in enumerate(self.reactions)
            ],
            float_format=float_format,
        )
        return self._create_pretty_table(**vars(constraints_table_options))

    def construct_reactions_table(self):
        reactions_table_options = TableOptions(
            headers=[
                "Reaction",
                "X",
                "Y",
                "Z",
                "Fx",
                "Fy",
                "Fz",
                "Mx",
                "My",
                "Mz",
            ],
            data=[
                [
                    reaction.name if reaction.name else f"Reaction {i+1}",
                    *reaction.location,
                    *reaction.magnitude,
                ]
                for i, reaction in enumerate(self.reactions)
            ],
            float_format="{:.2f}",
        )
        return self._create_pretty_table(**vars(reactions_table_options))

    # def _create_pretty_tables(self, float_format):
    #     # Load table
    #     loads_table = PrettyTable()
    #     loads_table.field_names = [
    #         "Load",
    #         "Loc X",
    #         "Loc Y",
    #         "Loc Z",
    #         "Fx",
    #         "Fy",
    #         "Fz",
    #         "Mx",
    #         "My",
    #         "Mz",
    #     ]
    #     for i, load in enumerate(self.loads):
    #         loc_x, loc_y, loc_z = load.location
    #         fx, fy, fz, mx, my, mz = load.magnitude
    #         loads_table.add_row(
    #             [
    #                 load.name if load.name else f"Load {i+1}",
    #                 float_format.format(loc_x),
    #                 float_format.format(loc_y),
    #                 float_format.format(loc_z),
    #                 float_format.format(fx),
    #                 float_format.format(fy),
    #                 float_format.format(fz),
    #                 float_format.format(mx),
    #                 float_format.format(my),
    #                 float_format.format(mz),
    #             ]
    #         )

    #     # Constraints table
    #     constraints_table = PrettyTable()
    #     constraints_table.field_names = ["Reaction", "Constraint Matrix"]
    #     for i, reaction in enumerate(self.reactions):
    #         constraint_matrix_str = "\n".join(
    #             [
    #                 "[" + " ".join(float_format.format(val) for val in row) + "]"
    #                 for row in reaction.constraint
    #             ]
    #         )
    #         constraints_table.add_row(
    #             [
    #                 reaction.name if reaction.name else f"Reaction {i+1}",
    #                 constraint_matrix_str,
    #             ]
    #         )

    #     # Reactions table
    #     reactions_table = PrettyTable()
    #     reactions_table.field_names = [
    #         "Reaction",
    #         "Loc X",
    #         "Loc Y",
    #         "Loc Z",
    #         "Fx",
    #         "Fy",
    #         "Fz",
    #         "Mx",
    #         "My",
    #         "Mz",
    #     ]
    #     for i, reaction in enumerate(self.reactions):
    #         loc_x, loc_y, loc_z = reaction.location
    #         row = [
    #             reaction.name if reaction.name else f"Reaction {i+1}",
    #             float_format.format(loc_x),
    #             float_format.format(loc_y),
    #             float_format.format(loc_z),
    #         ]
    #         row.extend(float_format.format(val) for val in reaction.magnitude)
    #         reactions_table.add_row(row)

    #     return loads_table, constraints_table, reactions_table

    def _print_table_with_colored_borders(
        self,
        table: str,
        vertical_border_color="\033[32m",
        horizontal_border_color="\033[90m",
    ) -> str:
        """Helper function to print table with green horizontal borders only."""
        reset_color = "\033[0m"
        table_str = table.get_string()
        table_new = []
        for line in table_str.splitlines():
            if set(line.strip()) <= {"+", "-", "="}:
                print(f"{vertical_border_color}{line}{reset_color}")
            else:
                colored_line = line.replace(
                    "|", f"{horizontal_border_color}|{reset_color}"
                )
                table_new.append(colored_line)

        return table_new


class AssemblyResult(ReactionResult):

    def __init__(self, assembly: Assembly, solution: List[float]):

        self.input = assembly
        self.solution = solution

        self.loads = self.extract_loads(solution)
        self.reactions = self.extract_reactions(solution)

    def _extract_attr_from_assembly(self, attr: str, assembly: Assembly) -> list:
        assembly = self.assembly if assembly is None else assembly
        return [item for part in assembly.parts for item in getattr(part, attr)]

    def extract_reactions(self, assembly: Assembly = None) -> List[Reaction]:
        return self._extract_attr_from_assembly("reactions", assembly)

    def extract_loads(self, assembly: Assembly = None) -> List[Load]:
        return self._extract_attr_from_assembly("loads", assembly)

    def test(self):
        pass
