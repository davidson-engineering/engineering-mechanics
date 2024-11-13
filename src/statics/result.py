from abc import abstractmethod
from datetime import datetime

from prettytable import PrettyTable

from .statics import logger


class Result:

    @property
    @abstractmethod
    def constants(self): ...

    @property
    @abstractmethod
    def equations(self): ...

    @abstractmethod
    def update_equations(self, solution): ...


class ReactionResult(Result):
    def __init__(self, reactions, loads, solution, report=None):
        self.loads = loads
        self.report = report
        self.reactions = self.update_equations(reactions, solution)

    def update_equations(self, reactions, solution):
        for i, reaction in enumerate(reactions):
            reaction.magnitude = solution[i]
        return reactions

    @property
    def constants(self):
        return self.loads

    @property
    def equations(self):
        return self.reactions

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
                    "[" + " ".join(float_format.format(val) for val in row) + "]"
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
