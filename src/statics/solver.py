from __future__ import annotations
import logging
from typing import List, Union

import numpy as np
from mechanics.assembly import Assembly, Connection, Part
from simulation.solver import LinearSolver, leastsquares_solver
from common.types import Reaction

logger = logging.getLogger(__name__)

from mechanics import Load


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


class ReactionSolver(LinearSolver):

    def __init__(
        self,
        loads: List[Load] = None,
        reactions: List[Reaction] = None,
    ):
        super().__init__(equations=reactions, constants=loads)
        if self.equations:
            self.check_constraints()

    @property
    def loads(self):
        return self.constants

    @property
    def reactions(self):
        return self.equations

    def construct_constant_vector(self, loads: List[Load] = None) -> np.ndarray:
        """Construct the right-hand side vector b for the equilibrium equations."""
        loads = loads or self.loads
        b = np.zeros(6)

        if loads is None:
            return b

        locations = np.array([load.location for load in loads])  # Shape (n_loads, 3)
        magnitudes = np.array([load.magnitude for load in loads])  # Shape (n_loads, 6)

        # Step 1: Force equilibrium - sum of external loads
        b = -np.sum(magnitudes, axis=0)

        moments_from_forces = np.cross(
            locations, magnitudes[:, 0:3]
        )  # Shape (n_loads, 3)
        # Sum all moments from forces
        b[3:6] -= np.sum(moments_from_forces, axis=0)

        # Add applied moments
        cross_offsets = np.cross(locations, magnitudes[:, 3:6])  # Shape (n_moments, 3)

        # Sum applied moments and cross products of moments with their locations
        b[3:6] -= np.sum(cross_offsets, axis=0)

        return b

    def construct_coeff_matrix(self, reactions: List[Reaction] = None) -> np.ndarray:
        """Construct the equilibrium matrix A for the reaction forces and moments."""

        reactions = reactions or self.reactions

        A = np.zeros((6, 6 * len(reactions)))

        for i, reaction in enumerate(reactions):
            reaction_index = 6 * i  # Column index for this reaction's components
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

    def check_constraints(self):

        def check_constraint_matrix_size(reaction):
            if reaction.constraint.shape not in [(6,), (6, 6)]:
                raise ValueError(
                    f"Constraint matrix for reaction '{reaction.name}' must be a 1x6 vector or a 6x6 matrix."
                )

        for reaction in self.reactions:
            check_constraint_matrix_size(reaction)


class AssemblySolver(ReactionSolver):
    def __init__(
        self,
        assembly: Union[Assembly, List[Part]],
        tolerance: float = 1e-2,
        iteration_limit: int = 1000,
        modifier: np.ndarray = np.ones(6),
    ):
        super().__init__()

        self.assembly = assembly
        self.tolerance = tolerance
        self.iteration_limit = iteration_limit
        self.residuals = []
        self.modifier = modifier
        self.preffered_method = leastsquares_solver

    def solve_part(self, part: Part):
        equations = [conn.master for conn in part.connections]
        constants = part.loads + [conn.master for conn in part.connections]
        A = self.construct_coeff_matrix(equations)
        b = self.construct_constant_vector(constants)

        # Solve the system of equations
        solution, _ = super().solve(A, b)

        self.update_part(part, solution)

    def update_part(self, part: Part, solution: np.ndarray):
        # iterate throguh rows of solution
        residuals = []
        for reaction, conn in zip(solution, part.connections):
            residual = reaction * self.modifier
            residuals.append(residual)
            residuals.append(residual)
            conn.master.magnitude += residual
            conn.slave.magnitude -= residual

        self.residuals.append(np.mean(np.abs(residuals), axis=0))

    def solve(self):
        while True:
            for part in self.assembly.parts:
                self.solve_part(part)
            if np.mean(self.residuals[-1]) < self.tolerance:
                logger.info(
                    f"Equilibrium reached in {len(self.residuals)} iterations.",
                    extra={
                        "residuals": self.residuals[-1],
                        "tolerance": self.tolerance,
                    },
                )
                break
            elif len(self.residuals) > self.iteration_limit:
                logger.error(
                    "Iteration limit reached.", extra={"residuals": self.residuals[-1]}
                )
                break


if __name__ == "__main__":
    from mechanics import Rod, Disc

    rod1 = Rod(id="rod", length=1, mass=1)
    rod_connect_R = Reaction(name=1, location=[1, 0, 0], constraint=np.eye(6))
    rod_ground = Reaction(name=0, location=[0, 0, 0], constraint=np.eye(6))
    ground = Reaction(name="ground", location=[0, 0, 0], constraint=np.eye(6))

    disc1 = Disc(id="disc", radius=0.5, mass=1)
    disc_connect_R = Reaction(name=2, location=[1, 0, 0], constraint=np.eye(6))
    rod_disc_connection = Connection(master=rod_connect_R, slave=disc_connect_R)
    rod_ground_connection = Connection(master=rod_ground, slave=ground)
    load = Load(name="disc_load", magnitude=[-10, 0, -10, 0, 0, 0], location=[2, 0, 0])

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

    print(assembly.parts)
    solver = AssemblySolver(assembly)

    reactions = solver.solve()
