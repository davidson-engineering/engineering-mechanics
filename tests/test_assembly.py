import pytest
import numpy as np
from mechanics import Rod, Disc
from base.assembly import Connection, Part, Assembly
from base.vector import Load, Reaction
from statics.solver import AssemblySolver
from statics.study import AssemblyStudy


@pytest.fixture
def assembly():
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
    return assembly


def test_assembly_study_initialization(assembly):

    study = AssemblyStudy(
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=assembly,
    )

    assert study.name == "Test Study"
    assert study.description == "Test study with one load and one reaction"
    assert study.assembly == assembly


def test_add_gravity_loads(assembly):

    study = AssemblyStudy(
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=assembly,
    )

    study.add_gravity_loads(gravity=[0, 0, -9.81])

    for part in study.assembly.parts:
        for load in part.loads:
            if load.name.endswith(" weight"):
                assert np.array_equal(load.magnitude, [0, 0, -9.81, 0, 0, 0])


def test_run(assembly):

    study = AssemblyStudy(
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=assembly,
    )

    solution = study.run()

    assert solution is not None
    # assert study.solver.report is not None


def test_extract_reactions(assembly):

    solver = AssemblySolver(
        assembly=assembly,
    )

    reactions = solver.extract_reactions()

    assert len(reactions) == 3
    assert reactions[0].name == 1

    loads = solver.extract_loads()

    assert len(loads) == 1
    assert loads[0].name == "disc_load"


def test_plot_assembly(assembly):

    from visual.plotly_plot import plot_loads_3d

    study = AssemblyStudy(
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=assembly,
    )

    study.add_gravity_loads(gravity=[0, 0, -9.81])

    solution = study.run()

    plot_loads_3d(
        loads=solution.loads,
        parts=solution.solution.parts,
        scale=1,
    )
