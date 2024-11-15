from typing import Any, List, Union
import numpy as np
from numpy.typing import ArrayLike
from common.types import Load
from mechanics.assembly import Assembly, Part
from mechanics.mechanics import Bodies, Body
from simulation.study import LinearStudy
from statics import AssemblySolver, Reaction, ReactionSolver
from statics.result import AssemblyResult, ReactionResult


class StaticsStudy(LinearStudy):

    _result_factory = ReactionResult

    def __init__(
        self,
        name: str,
        description: str,
        reactions: List[Reaction],
        loads: List[Load] = None,
        bodies: List[Union[Body, Bodies]] = None,
        study_id=None,
        gravity: ArrayLike = [0, 0, -9.81],
    ):

        self.bodies = bodies
        self.loads = [] if loads is None else loads
        self.reactions = reactions
        self.gravity = gravity

        if gravity is not None and bodies:
            self.add_gravity_loads(gravity=gravity)

        super().__init__(
            name,
            description,
            study_id,
            solver=ReactionSolver(loads=self.loads, reactions=self.reactions),
        )

    def add_gravity_loads(
        self, gravity: Union[list, ArrayLike] = [0, 0, -9.81]
    ) -> None:
        """Add gravity loads to the study."""

        for body in self.bodies:
            self.loads.append(
                Load(
                    magnitude=body.mass * np.asarray(gravity),
                    location=body.cog,
                    name=f"{body.id} weight",
                )
            )

    def run(self):
        solution, report = self.solver.solve()
        result = self.build_result(
            reactions=self.reactions, loads=self.loads, solution=solution, report=report
        )
        self.validate_result(result)
        return result


class AssemblyStudy(LinearStudy):

    _result_factory = AssemblyResult

    def __init__(
        self,
        name,
        description,
        assembly,
        study_id=None,
        gravity: ArrayLike = [0, 0, -9.81],
    ):
        self.assembly = assembly
        super().__init__(
            name, description, study_id, solver=AssemblySolver(self.assembly)
        )
        if gravity is not None:
            self.add_gravity_loads(gravity=gravity)

    def run(self):
        self.solver.solve()
        print(self.assembly)

    def add_gravity_loads(
        self, gravity: Union[list, ArrayLike] = [0, 0, -9.81]
    ) -> None:
        """Add gravity loads to the study."""

        for part in self.assembly.parts:
            if isinstance(part.bodies, list):
                bodies = Bodies(bodies=part.bodies)
            elif isinstance(part.bodies, Body):
                bodies = part.bodies
            else:
                raise ValueError("Invalid bodies type in part.")

            part.loads.append(
                Load(
                    magnitude=bodies.mass * np.asarray(gravity),
                    location=bodies.cog,
                    name=f"{part.id} weight",
                )
            )


def plot_residuals(residuals):
    import matplotlib.pyplot as plt

    plt.plot(residuals)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Residual")
    plt.title("Mean Residual vs. Iteration")
    plt.show()


if __name__ == "__main__":
    from mechanics import Rod, Disc
    from mechanics.assembly import Connection
    import logging

    logging.basicConfig(level=logging.INFO)

    rod1 = Rod(id="rod", length=1, mass=1)
    rod_connect_R = Reaction(name=1, location=[1, 0, 0], constraint=np.eye(6))
    rod_ground = Reaction(name=0, location=[0, 0, 0], constraint=np.eye(6))
    ground = Reaction(name="ground", location=[0, 0, 0], constraint=np.eye(6))

    disc1 = Disc(id="disc", radius=0.5, mass=1)
    disc_connect_R = Reaction(name=2, location=[1, 0, 0], constraint=np.eye(6))
    rod_disc_connection = Connection(master=rod_connect_R, slave=disc_connect_R)
    rod_ground_connection = Connection(master=rod_ground, slave=ground)
    load = Load(name="disc_load", magnitude=[-10, 0, -10, 23, 0, 0], location=[2, 0, 0])

    rod_part = Part(
        id="rod_part",
        bodies=[rod1],
        connections=[rod_disc_connection, rod_ground_connection],
    )
    disc_part = Part(
        id="disc_part",
        bodies=[disc1],
        connections=[rod_disc_connection.invert()],
        loads=[load],
    )

    assembly = Assembly(parts=[rod_part, disc_part])

    study = AssemblyStudy(
        name="Example Study",
        description="Example study with one load and one reaction",
        assembly=assembly,
        gravity=[0, 0, -9.81],
    )

    study.run()
    plot_residuals(study.solver.residuals)
