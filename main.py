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

from statics import StaticsCalculator, BoundVector, Reaction


def main():

    import numpy as np

    # Define forces and reactions here, then use StaticsCalculator to analyze
    forces = [
        BoundVector(magnitude=np.array([0, 0, -100]), location=np.array([1, 0, 0]))
    ]
    moments = [
        BoundVector(magnitude=np.array([0, 10, -5]), location=np.array([0, 1, 0]))
    ]
    reactions = [Reaction(location=np.array([0, 0, 0]), constraint=np.eye(6))]

    calculator = StaticsCalculator(forces, moments, reactions)
    reactions_result = calculator.solve_reactions()

    print(f"{reactions_result=}")


if __name__ == "__main__":
    main()
