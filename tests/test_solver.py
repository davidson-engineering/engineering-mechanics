import pytest
import numpy as np

from base.assembly import Assembly, Part, Connection
from mechanics.mechanics import Rod, Disc
from base.vector import Load, Reaction
from statics.solver import AssemblySolver


def test_assembly_solver_initialization():
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
    solver = AssemblySolver(assembly)

    assert solver.assembly == assembly
    assert solver.tolerance == 1e-6
    assert solver.iteration_limit == 1000
    assert np.array_equal(solver.modifier, np.ones(6))


def test_solve_part():
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
    solver = AssemblySolver(assembly)

    solver.solve_part(disc_part)

    assert len(solver.residuals) > 0


def test_solve():
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
    solver = AssemblySolver(assembly)

    solver.solve()

    assert len(solver.residuals) > 0
    assert np.mean(solver.residuals[-1]) < solver.tolerance
