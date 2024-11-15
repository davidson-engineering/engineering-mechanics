import pytest
import numpy as np
from mechanics import Rod, Disc
from mechanics.assembly import Connection, Part, Assembly
from common.types import Load, Reaction
from statics.study import AssemblyStudy


def test_assembly_study_initialization():
    rod1 = Rod(id="rod", length=1, mass=1)
    disc1 = Disc(id="disc", radius=0.5, mass=1)

    rod_connect_R = Reaction(name=1, location=[1, 0, 0], constraint=np.eye(6))
    rod_ground = Reaction(name=0, location=[0, 0, 0], constraint=np.eye(6))
    ground = Reaction(name="ground", location=[0, 0, 0], constraint=np.eye(6))

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
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=assembly,
        gravity=[0, 0, -9.81],
    )

    assert study.name == "Test Study"
    assert study.description == "Test study with one load and one reaction"
    assert study.assembly == assembly


def test_add_gravity_loads():
    rod1 = Rod(id="rod", length=1, mass=1)
    disc1 = Disc(id="disc", radius=0.5, mass=1)

    rod_connect_R = Reaction(name=1, location=[1, 0, 0], constraint=np.eye(6))
    rod_ground = Reaction(name=0, location=[0, 0, 0], constraint=np.eye(6))
    ground = Reaction(name="ground", location=[0, 0, 0], constraint=np.eye(6))

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
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=assembly,
        gravity=[0, 0, -9.81],
    )

    study.add_gravity_loads(gravity=[0, 0, -9.81])

    for part in study.assembly.parts:
        for load in part.loads:
            assert np.array_equal(load.magnitude, [0, 0, -9.81])
            assert load.name.endswith(" weight")


def test_run():
    rod1 = Rod(id="rod", length=1, mass=1)
    disc1 = Disc(id="disc", radius=0.5, mass=1)

    rod_connect_R = Reaction(name=1, location=[1, 0, 0], constraint=np.eye(6))
    rod_ground = Reaction(name=0, location=[0, 0, 0], constraint=np.eye(6))
    ground = Reaction(name="ground", location=[0, 0, 0], constraint=np.eye(6))

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
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=assembly,
        gravity=[0, 0, -9.81],
    )

    study.run()

    assert study.solver.solution is not None
    assert study.solver.report is not None
