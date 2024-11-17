import pytest
import numpy as np
from base.vector import BoundVector

from base.plotting import (
    _convert_unit_vector,
    generate_hover_text,
    build_load_traces,
    build_force_traces,
    build_moment_traces,
)


def test_convert_unit_vector():
    vector = np.array([3, 4, 0])
    expected = np.array([0.6, 0.8, 0])
    result = _convert_unit_vector(vector)
    np.testing.assert_array_almost_equal(result, expected)

    zero_vector = np.array([0, 0, 0])
    expected_zero = np.array([0, 0, 0])
    result_zero = _convert_unit_vector(zero_vector)
    np.testing.assert_array_almost_equal(result_zero, expected_zero)


def test_generate_hover_text():
    name = "Test Vector"
    magnitude = np.array([1, 2, 3, 4, 5, 6])
    expected = "Test Vector<br>Fx: 1.00 N<br>Fy: 2.00 N<br>Fz: 3.00 N<br>Mx: 4.00 Nm<br>My: 5.00 Nm<br>Mz: 6.00 Nm"
    result = generate_hover_text(name, magnitude)
    assert result == expected

    magnitude_zero_force = np.array([0, 0, 0, 4, 5, 6])
    expected_zero_force = "Test Vector<br>Fx: 0.00 N<br>Fy: 0.00 N<br>Fz: 0.00 N<br>Mx: 4.00 Nm<br>My: 5.00 Nm<br>Mz: 6.00 Nm"
    result_zero_force = generate_hover_text(name, magnitude_zero_force)
    assert result_zero_force == expected_zero_force

    magnitude_zero_moment = np.array([1, 2, 3, 0, 0, 0])
    expected_zero_moment = "Test Vector<br>Fx: 1.00 N<br>Fy: 2.00 N<br>Fz: 3.00 N"
    result_zero_moment = generate_hover_text(name, magnitude_zero_moment)
    assert result_zero_moment == expected_zero_moment


def test_build_load_traces():
    location = np.array([0, 0, 0])
    magnitude = np.array([1, 2, 3, 4, 5, 6])
    item = BoundVector(location=location, magnitude=magnitude)
    traces = build_load_traces(
        item, name="Test Vector", scale=1.0, as_components=False, color="blue"
    )
    assert len(traces) > 0  # Ensure traces are created


def test_build_force_traces():
    location = np.array([0, 0, 0])
    magnitude = np.array([1, 2, 3])
    traces = build_force_traces(
        location, magnitude, name="Test Vector", scale=1.0, color="blue"
    )
    assert len(traces) == 2  # One body and one tip


def test_build_moment_traces():
    location = np.array([0, 0, 0])
    moments = np.array([1, 2, 3])
    traces = build_moment_traces(
        location, moments, scale=1.0, name="Test Moments", color="red"
    )
    assert len(traces) > 0  # Ensure traces are created
