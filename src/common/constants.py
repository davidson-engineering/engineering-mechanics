import numpy as np

# Define gravitational acceleration in m/s^2
GRAVITY = np.array([0, 0, -9.81])

# Define conversion factors for force and moment units
UNIT_CONVERSIONS = {
    "unitless": 1,
    "N": 1,
    "kN": 1e3,
    "MN": 1e6,
    "kg": 1,  # base unit for mass
    "g": 1e-3,  # grams to kilograms
    "ton": 1e3,  # metric tons to kilograms
    "Nm": 1,  # base unit for moments in Newton-meters
    "Nmm": 1e-3,  # Newton-millimeters to Newton-meters
    "kNm": 1e3,  # kilonewton-meters to Newton-meters
}

PLOT_AXES = [
    {"axis": "X", "direction": [1, 0, 0], "normal": [0, 1, 0]},  # Rotation about X
    {"axis": "Y", "direction": [0, 1, 0], "normal": [0, 0, 1]},  # Rotation about Y
    {"axis": "Z", "direction": [0, 0, 1], "normal": [1, 0, 0]},  # Rotation about Z
]
