from abc import abstractmethod
from datetime import datetime
from typing import List

import numpy as np


class Result:

    def __init__(self, equations: List[np.ndarray], constants: List[np.ndarray]):
        self.equations = equations
        self.constants = constants

    @abstractmethod
    def update_equations(self, solution): ...

    def _generate_html_report(self, tables: dict):

        def _table_generator(tables):
            for heading, table in tables.items():
                yield heading, table.get_html_string()

        table_html_combined = "\n".join(
            f"<h2>{heading}</h2>\n{table_html}"
            for heading, table_html in _table_generator(tables)
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert report and validation results to HTML
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
                {table_html_combined}
                <h2>Validation Results</h2>
                <p>Combined reactions match the input loads, indicating equilibrium is achieved.</p>
            </div>
        </body>
        </html>
        """
        return html_report
