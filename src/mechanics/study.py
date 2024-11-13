from typing import List, Union

from common.types import Load
from mechanics.mechanics import Bodies, Body
from simulation.study import LinearStudy


class MechanicsStudy(LinearStudy):

    def __init__(
        self,
        name: str,
        description: str,
        bodies: Union[Bodies, List[Body]],
        loads: List[Load],
        study_id=None,
        solver=None,
    ):
        super().__init__(name, description, study_id, solver)
        self.bodies = bodies
        self.loads = loads