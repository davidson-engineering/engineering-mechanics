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

__version__ = "0.7.0"  # the package version: pyproject.toml reads it from here

Revolute = Pin
Prismatic = Slider
Spherical = Ball


def load(path) -> Model:
    """Load a model from a YAML file."""
    from engmech.io.loader import load_model

    return load_model(path)


def loads(text: str) -> Model:
    """Load a model from YAML text."""
    from engmech.io.loader import loads_model

    return loads_model(text)


def mass_properties(*shapes, units=None) -> MassProperties:
    """Combined mass properties of shapes, without building a model.

    >>> p = mass_properties(Box(mass=2, size=[1, 1, 1], center=[0, 0, 0]))
    >>> p.mass, p.cog, p.inertia
    """
    from engmech import mass as mp
    from engmech.inputs import Resolver
    from engmech.units import Context

    r = Resolver(Context(UnitSystem.from_spec(units)))
    return mp.combine([s.resolve(r, f"shapes[{i}]") for i, s in enumerate(shapes)])


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
    "loads",
    "mass_properties",
]
