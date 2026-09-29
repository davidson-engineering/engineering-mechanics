"""engmech: rigid-body statics, dynamics and mass properties.

Quick start::

    import engmech as em

    m = em.Model("Simply supported beam", planar=True, units={"length": "m", "force": "kN"})
    m.support("A", em.Pin(at=[0, 0]))
    m.support("B", em.Roller(at=[6, 0], normal="+y"))
    m.load(em.Force([0, -12], at=[2, 0]))
    result = m.solve()
    result.show()
    result.primary["A"].force   # SI (N)

Models can also be written as YAML files and loaded with :func:`load`.
"""

from engmech.errors import EngmechError, InputError
from engmech.joints import (
    Ball,
    Bearing,
    Cable,
    Contact,
    Custom,
    Cylindrical,
    Fixed,
    Link,
    Pin,
    Roller,
    Slider,
    Universal,
)
from engmech.loads import DistributedLoad, Force, Moment, Motion, UnknownLoad
from engmech.mass import MassProperties
from engmech.model import Check, Model
from engmech.shapes import Box, Cone, CustomMass, Cylinder, PointMass, Rod, Sphere
from engmech.units import UnitSystem

__version__ = "0.2.0"

Revolute = Pin
Prismatic = Slider
Spherical = Ball


def load(path) -> Model:
    """Load a model from a YAML file."""
    from engmech.io.loader import load_model

    return load_model(path)


__all__ = [
    "Ball",
    "Bearing",
    "Box",
    "Cable",
    "Check",
    "Cone",
    "Contact",
    "Custom",
    "CustomMass",
    "Cylinder",
    "Cylindrical",
    "DistributedLoad",
    "EngmechError",
    "Fixed",
    "Force",
    "InputError",
    "Link",
    "MassProperties",
    "Model",
    "Moment",
    "Motion",
    "Pin",
    "PointMass",
    "Prismatic",
    "Revolute",
    "Rod",
    "Roller",
    "Slider",
    "Sphere",
    "Spherical",
    "UnitSystem",
    "Universal",
    "UnknownLoad",
    "load",
]
