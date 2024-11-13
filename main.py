#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2024-01-01
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Example usage of the statics module."""
# ---------------------------------------------------------------------------

# Application confugration not currently used
# from logging.config import dictConfig

# # Import the load_configs function
# from config_loader import load_configs
# LOGGING_CONFIG_FILEPATH = "config/logging.yaml"
# APP_CONFIG_FILEPATH = "config/application.toml"

# # Load user configurations using the config_loader module
# configs = load_configs([APP_CONFIG_FILEPATH, LOGGING_CONFIG_FILEPATH])

# # Configure logging using the specified logging configuration
# dictConfig(configs["logging"])

from statics import ReactionSolver, Load, Reaction, StaticsStudy
from mechanics import Body


def main():

    import numpy as np

    bodies = [
        Body(
            id="Cuboid",
            mass=1,
            inertia=np.diag([1 / 12, 1 / 12, 1 / 12]),
        )
    ]
    loads = [
        Load(
            magnitude=np.array([0, 0, -1100.776, 0, -10, 0]),
            location=np.array([3.1053, 0, 0]),
            name="F_a",
        )
    ]
    reactions = [
        Reaction(
            location=np.array([0, 0, 0]), constraint=np.eye(6), name="Fixed support"
        ),
    ]

    # Define forces and reactions here, then use StaticsCalculator to analyze
    study = StaticsStudy(
        name="Example Study",
        description="Example study with one load and one reaction",
        reactions=reactions,
        loads=loads,
        bodies=bodies,
        gravity=[0, 0, -9.81],
    )
    result = study.run()
    result.print_summary(html_report_path="report.html")


if __name__ == "__main__":
    main()
