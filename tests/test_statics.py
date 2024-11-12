import pytest
import numpy as np

from mechanics.mechanics import Bodies
from statics import (
    ReactionSolver,
    BoundVector,
    Reaction,
    Load,
    UnderconstrainedError,
    IllConditionedError,
    OverconstrainedWarning,
)
from constants.constants import GRAVITY
from mechanics import Body


def test_mass_to_force_conversion():
    # Test that a mass converts correctly to a force with gravity applied
    mass = Body(mass=10, cog=np.array([1, 1, 1]))
    force = mass * GRAVITY

    # Check the force is downward with magnitude 10 * 9.81 N
    expected_force = np.array([0, 0, -98.1])
    assert np.allclose(
        force.magnitude, expected_force
    ), "Mass to force conversion failed"
    assert np.array_equal(
        force.location, np.array([1, 1, 1])
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
    reactions_result = solver.solve_reactions()

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
    reactions_result = calculator.solve_reactions()

    # Expected reaction: upward force balancing the mass (10 * 9.81 N)
    expected_reaction = np.array([0, 0, 98.1, 0, 0, 0])
    assert np.allclose(
        reactions_result[0], expected_reaction
    ), "Statics calculation with mass failed"


def _test_statics_calculator_func(loads, reactions, expected_reactions):
    # Instantiate the statics calculator
    calculator = ReactionSolver(loads=loads, reactions=reactions)
    reactions_result = calculator.solve_reactions()

    # Check each reaction matches the expected result
    for i, expected_reaction in enumerate(expected_reactions):
        assert np.allclose(
            reactions_result[i], expected_reaction
        ), f"Reaction {i+1} does not match expected value"


def test_statics_calculator_multiple_reactions_1():
    # Forces applied at different locations
    loads = [
        BoundVector(magnitude=np.array([0, 0, -100]), location=np.array([0, 0, 1])),
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

    _test_statics_calculator_func(loads, reactions, expected_reactions)


def test_statics_calculator_multiple_reactions_2():
    # Forces applied at different locations
    loads = [
        BoundVector(magnitude=np.array([0, 0, -100]), location=np.array([0, 0, 1])),
        BoundVector(magnitude=np.array([100, 0, 0]), location=np.array([0, 0, 0])),
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

    _test_statics_calculator_func(loads, reactions, expected_reactions)


def test_statics_calculator_multiple_reactions_3():
    # Forces applied at different locations
    loads = [
        BoundVector(magnitude=np.array([0, 0, -150]), location=np.array([0, 0, 1])),
        BoundVector(magnitude=np.array([100, 0, 0]), location=np.array([0, 0, 0])),
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

    _test_statics_calculator_func(loads, reactions, expected_reactions)


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
        _test_statics_calculator_func(loads, reactions, None)

    loads.append(
        BoundVector(
            magnitude=np.array([0, 0, 0, 100, 0, 0]), location=np.array([0, 0, 0])
        )
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

    _test_statics_calculator_func(loads, reactions, expected_reactions)


def test_statics_calculator_multiple_reactions_5():

    body_1 = Body(mass=10, cog=np.array([3, 0, 1]))
    body_2 = Body(mass=100, cog=np.array([-3, 0.5, 1]))
    bodies = Bodies(bodies=[body_1, body_2])

    assert bodies.mass == 110

    loads = [bodies * GRAVITY]

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
    _test_statics_calculator_func(loads, reactions, expected_reactions)
