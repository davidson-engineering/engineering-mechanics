from abc import abstractmethod
from typing import List

import numpy as np


class Result:

    def __init__(self, equations: List[np.ndarray], constants: List[np.ndarray]):
        self.equations = equations
        self.constants = constants

    @abstractmethod
    def update_equations(self, solution): ...
