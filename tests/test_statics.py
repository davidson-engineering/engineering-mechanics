import pytest
import numpy as np

from mechanics.mechanics import Bodies
from statics import (
    ReactionSolver,
    Reaction,
    Load,
)
from base.solver import (
    UnderconstrainedError,
    IllConditionedError,
    OverconstrainedWarning,
)
from common.constants import GRAVITY
from mechanics import Body
from statics.study import StaticsStudy


@pytest.fixture
def sample_loads():
    # Define sample loads with positions and magnitudes
    load1 = Load(
        location=np.array([1.0, 0.0, 0.0]),
        magnitude=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    load2 = Load(
        location=np.array([0.0, 1.0, 0.0]),
        magnitude=np.array([0.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
    )
    return [load1, load2]


@pytest.fixture
def sample_reactions():
    # Define sample reactions with locations and constraints
    reaction1 = Reaction(
        location=np.array([0.0, 0.0, 0.0]), constraint=np.array([1, 1, 1, 1, 1, 0])
    )
    reaction2 = Reaction(
        location=np.array([0.0, 1.0, 0.0]), constraint=np.array([1, 1, 1, 1, 0, 0])
    )
    return [reaction1, reaction2]


@pytest.fixture
def reaction_solver(sample_loads, sample_reactions):
    return ReactionSolver(loads=sample_loads, reactions=sample_reactions)


@pytest.fixture
def sample_study(sample_loads, sample_reactions):
    return StaticsStudy(
        name="Test Study",
        description="Test study description",
        bodies=None,
        loads=sample_loads,
        reactions=sample_reactions,
    )


### Test Cases


def test_assemble_equilibrium_matrix(reaction_solver):
    """Test that the equilibrium matrix (A) and load vector (b) are correctly assembled."""
    A = reaction_solver.construct_coeff_matrix()
    b = reaction_solver.construct_constant_vector()

    # Check dimensions of A and b
    assert A.shape == (6, 6 * len(reaction_solver.reactions))
    assert b.shape == (6,)


def test_solve_reactions(reaction_solver):
    """Test that reactions are solved without errors and that the solution is in equilibrium."""
    solution, _ = reaction_solver.solve()

    # Ensure a solution was found
    assert solution is not None
    assert solution.shape == (
        len(reaction_solver.reactions),
        6,
    )


def test_validate_reactions_passes(reaction_solver, sample_study):
    """Test that validate_reactions passes when reactions match input loads."""
    # Solve reactions and validate the solution
    solution, _ = sample_study.solve()
    result = sample_study.build_result(
        sample_study.reactions, sample_study.loads, solution
    )

    sample_study.validate_result(result)  # Should pass without error


def test_ill_conditioned_error(reaction_solver):
    """Test that an IllConditionedError is raised for an ill-conditioned matrix."""
    reaction_solver.reactions[0].constraint = np.zeros((6, 6))  # Make matrix singular
    reaction_solver.reactions[1].constraint = np.zeros((6, 6))  # Make matrix singular

    with pytest.raises(IllConditionedError):
        A = reaction_solver.construct_coeff_matrix()
        reaction_solver.check_condition_number(A)


def test_underconstrained_error(reaction_solver):
    """Test that an UnderconstrainedError is raised when the system is under-constrained."""
    reaction_solver.reactions.pop()  # Remove one reaction to under-constrain

    with pytest.raises(UnderconstrainedError):
        reaction_solver.solve()


# def test_overconstrained_warning(reaction_solver):
#     """Test that an OverconstrainedWarning is issued when the system is over-constrained."""
#     reaction_solver.reactions.append(
#         Reaction(location=np.array([0.0, 0.0, 0.0]), constraint=np.eye(6))
#     )  # Add redundant constraint

#     with pytest.warns(OverconstrainedWarning):
#         reaction_solver.solve()


# def test_print_summary(reaction_solver, capsys):
#     """Test that the summary printout works correctly."""
#     reactions_result = reaction_solver.solve()
#     reaction_solver.print_summary()

#     # Capture and check the printed output
#     captured = capsys.readouterr()
#     assert "Input loads Summary" in captured.out
#     assert "Constraints Summary" in captured.out
#     assert "Reactions Summary" in captured.out


# def test_html_report_generation(reaction_solver, tmp_path):
#     """Test that an HTML report is generated correctly."""
#     reactions_result = reaction_solver.solve()
#     html_report_path = tmp_path / "statics_solver_report.html"
#     reaction_solver.print_summary(html_report_path=html_report_path)

#     # Check that the HTML file was created and contains expected content
#     assert html_report_path.exists()
#     with open(html_report_path, "r") as f:
#         content = f.read()
#     assert "<html>" in content
#     assert "<h2>Input Loads Summary</h2>" in content
#     assert "<h2>Constraints Summary</h2>" in content
#     assert "<h2>Reactions Summary</h2>" in content


def test_mass_to_force_conversion():
    # Test that a mass converts correctly to a force with gravity applied
    mass = Body(mass=10, cog=np.array([1, 1, 1]))
    load = mass * GRAVITY

    # Check the force is downward with magnitude 10 * 9.81 N
    expected_load = np.array([0, 0, -98.1, 0, 0, 0])
    assert np.allclose(load.magnitude, expected_load), "Mass to force conversion failed"
    assert np.array_equal(
        load.location, np.array([1, 1, 1])
    ), "Force location incorrect after conversion"


def test_statics_calculator_simple():
    # Simple case: a single force balanced by a single fixed reaction
    loads = [
        Load(magnitude=np.array([0, 0, -100, 0, 0, 0]), location=np.array([0, 0, 0]))
    ]
    reactions = [
        Reaction(
            location=np.array([0, 0, 0]),
            constraint=np.array([1, 1, 1, 1, 1, 1]),  # Fixed constraint as 1x6 vector
        )
    ]

    solver = ReactionSolver(loads=loads, reactions=reactions)
    reactions_result = solver.solve()

    # Check that the reaction force balances the applied force
    expected_reaction = np.array([0, 0, 100, 0, 0, 0])  # Reaction in opposite direction
    assert np.allclose(
        reactions_result[0], expected_reaction
    ), "Simple statics calculation failed"


def test_statics_calculator_with_mass():
    # Case with mass: mass should be converted to force
    body = Body(mass=10, cog=np.array([1, 0, 0]))
    loads = [body * GRAVITY]
    reactions = [
        Reaction(
            location=np.array([1, 0, 0]),
            constraint=np.eye(6),  # Fixed constraint as 6x6 matrix
        )
    ]

    calculator = ReactionSolver(loads=loads, reactions=reactions)
    solution, _ = calculator.solve()

    # Expected reaction: upward force balancing the mass (10 * 9.81 N)
    expected_reaction = np.array([0, 0, 98.1, 0, 0, 0])
    assert np.allclose(
        solution[0], expected_reaction
    ), "Statics calculation with mass failed"


def _test_statics_calculator_func(
    reactions, loads=None, expected_reactions=None, bodies=None
):
    # Instantiate the statics calculator
    study = StaticsStudy(
        name="Test Study",
        description="Test study description",
        bodies=bodies,
        loads=loads,
        reactions=reactions,
        gravity=GRAVITY,
    )
    result = study.run()

    if expected_reactions is not None:
        # Check each reaction matches the expected result
        for i, expected_reaction in enumerate(expected_reactions):
            assert np.allclose(
                result.reactions[i].magnitude, expected_reaction
            ), f"Reaction {i+1} does not match expected value"


def test_statics_calculator_multiple_reactions_1():
    # Forces applied at different locations
    loads = [
        Load(magnitude=np.array([0, 0, -100]), location=np.array([0, 0, 1])),
    ]

    # Reactions: fixed and pinned constraints
    reactions = [
        Reaction(
            location=np.array([1, 0, 0]),
            constraint=np.eye(6),  # Fixed constraint
        ),
        Reaction(
            location=np.array([-1, 0, 0]),
            constraint=np.eye(6),  # Fixed constraint
        ),
    ]

    # Expected reactions for balancing forces and moments
    expected_reactions = [
        np.array([0, 0, 50, 0, 0, 0]),  # Reaction at the fixed point
        np.array([0, 0, 50, 0, 0, 0]),  # Reaction at the other fixed point
    ]

    _test_statics_calculator_func(
        loads=loads, reactions=reactions, expected_reactions=expected_reactions
    )


def test_statics_calculator_multiple_reactions_2():
    # Forces applied at different locations
    loads = [
        Load(magnitude=np.array([0, 0, -100]), location=np.array([0, 0, 1])),
        Load(magnitude=np.array([100, 0, 0]), location=np.array([0, 0, 0])),
    ]

    # Reactions: fixed, roller constraints with a mix of 1x6 and 6x6 constraints
    reactions = [
        Reaction(
            location=np.array([1, 0, 0]),
            constraint=np.eye(6),  # Fixed constraint as 6x6 matrix
        ),
        Reaction(
            location=np.array([-1, 0, 0]),
            constraint=np.array([0, 1, 1, 1, 1, 1]),  # Roller constraint as 1x6 vector
        ),
    ]

    # Expected reactions based on equilibrium
    expected_reactions = [
        np.array([-100, 0, 50, 0, 0, 0]),  # Reaction at the fixed point
        np.array([0, 0, 50, 0, 0, 0]),  # Reaction at the roller point
    ]

    _test_statics_calculator_func(
        loads=loads, reactions=reactions, expected_reactions=expected_reactions
    )


def test_statics_calculator_multiple_reactions_3():
    # Forces applied at different locations
    loads = [
        Load(magnitude=np.array([0, 0, -150]), location=np.array([0, 0, 1])),
        Load(magnitude=np.array([100, 0, 0]), location=np.array([0, 0, 0])),
    ]

    # Reactions: fixed, roller constraints with a mix of 1x6 and 6x6 constraints
    reactions = [
        Reaction(
            location=np.array([1, 0, 0]),
            constraint=np.eye(6),  # Fixed constraint as 6x6 matrix
        ),
        Reaction(
            location=np.array([-1, 0, 0]),
            constraint=np.array([0, 0, 1, 0, 0, 0]),  # Roller constraint as 1x6 vector
        ),
        Reaction(
            location=np.array([0, 0, 0]),
            constraint=np.array([0, 0, 0, 0, 0, 0]),  # Roller constraint as 1x6 vector
        ),
    ]

    # Expected reactions based on equilibrium
    expected_reactions = [
        np.array([-100, 0, 75, 0, 0, 0]),  # Reaction at the fixed point
        np.array([0, 0, 75, 0, 0, 0]),  # Reaction at the roller point
        np.array([0, 0, 0, 0, 0, 0]),  # Reaction at the roller point
    ]

    _test_statics_calculator_func(
        loads=loads, reactions=reactions, expected_reactions=expected_reactions
    )


def test_statics_calculator_multiple_reactions_4():

    body = Body(mass=10, cog=np.array([1, 0, 0]))
    loads = [body * [0, 0, -10]]

    # Reactions: fixed, roller constraints with a mix of 1x6 and 6x6 constraints
    reactions = [
        Reaction(
            location=np.array([1, 0, 0]),
            constraint=np.array([1, 1, 1, 0, 1, 1]),  # Fixed constraint with Mx free
        ),
        Reaction(
            location=np.array([-1, 0, 0]),
            constraint=np.array(
                [0, 1, 1, 0, 1, 1]
            ),  # Roller constraint, Fx and Mx free
        ),
    ]
    # Should raise an error due to underconstrained system about the x-axis
    with pytest.raises(UnderconstrainedError):
        _test_statics_calculator_func(loads=loads, reactions=reactions)

    loads.append(
        Load(magnitude=np.array([0, 0, 0, 100, 0, 0]), location=np.array([0, 0, 0]))
    )

    # Reactions: fixed, roller constraints with a mix of 1x6 and 6x6 constraints
    reactions = [
        Reaction(
            location=np.array([1, 0, 0]),
            constraint=np.array([1, 1, 1, 1, 1, 1]),  # Fixed constraint with Mx free
        ),
        Reaction(
            location=np.array([-1, 0, 0]),
            constraint=np.array(
                [0, 1, 1, 0, 1, 1]
            ),  # Roller constraint, Fx and Mx free
        ),
    ]

    # Expected Mx reaction on the fixed point only
    expected_reactions = [
        np.array([0, 0, 75, -100, -25, 0]),  # Reaction at the fixed point
        np.array([0, 0, 25, 0, -25, 0]),  # Reaction at the roller point
    ]

    _test_statics_calculator_func(
        loads=loads, reactions=reactions, expected_reactions=expected_reactions
    )


def test_statics_calculator_multiple_reactions_5():

    body_1 = Body(mass=10, cog=np.array([3, 0, 1]))
    body_2 = Body(mass=100, cog=np.array([-3, 0.5, 1]))
    bodies = [body_1, body_2]

    # Reactions: fixed, roller constraints with a mix of 1x6 and 6x6 constraints
    reactions = [
        Reaction(
            location=np.array([1, 0.5, 0]),
            constraint=np.array([0, 1, 1, 1, 1, 1]),  # Slider constraint as 1x6 vector
        ),
        Reaction(
            location=np.array([1, -0.5, 0]),
            constraint=np.eye(6),  # Fixed constraint as 6x6 matrix
        ),
        Reaction(
            location=np.array([-1, 0.5, 0]),
            constraint=np.array([0, 1, 1, 1, 1, 1]),  # Roller constraint as 1x6 vector
        ),
        Reaction(
            location=np.array([-1, -0.5, 0]),
            constraint=np.array([0, 1, 1, 1, 1, 1]),  # Roller constraint as 1x6 vector
        ),
    ]
    expected_reactions = np.array(
        [
            [0.0, 0.0, -12.2625, 98.1, 331.0875, 0.0],
            [0.0, 0.0, -110.3625, 98.1, 331.0875, 0.0],
            [0.0, 0.0, 649.9125, 98.1, 331.0875, 0.0],
            [0.0, 0.0, 551.8125, 98.1, 331.0875, 0.0],
        ]
    )

    # with pytest.raises(ValueError):
    _test_statics_calculator_func(
        reactions=reactions, expected_reactions=expected_reactions, bodies=bodies
    )
