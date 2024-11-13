from abc import abstractmethod
import logging
import numpy as np
from simulation.result import Result
from simulation.solver import LinearSolver

logger = logging.getLogger("simulation")


class Study:

    _result_factory = Result

    def __init__(self, name, description, study_id=None, solver=None):
        self.name = name
        self.description = description
        self.id = study_id
        self.solver = solver
        self.results = None

    def __str__(self):
        return f"{self.name} - {self.description}"

    def build_result(self, coefficients, constants, solution, report=None):
        return self._result_factory(coefficients, constants, solution, report)

    @abstractmethod
    def validate_result(self, result): ...

    def solve(self):
        return self.solver.solve()


class LinearStudy(Study):

    def __init__(self, name, description, study_id=None, solver=None):
        super().__init__(name, description, study_id, solver)

    def validate_result(self, result):
        """Validate the reaction results by checking equilibrium."""
        assert isinstance(result, self._result_factory)
        b = self.solver.construct_constant_vector(result.constants)
        b_equations = self.solver.construct_constant_vector(result.equations)
        if np.allclose(b, -b_equations):
            logger.info("Equilibrium check passed.")
            return True
        else:
            logger.error("Equilibrium check failed.")
            return False
