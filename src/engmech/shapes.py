"""Shape specifications for building mass properties from raw input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from engmech import mass as mp
from engmech.errors import InputError
from engmech.inputs import Resolver

# kg/m³: from aerogels and foams to the densest elements (osmium 22 590)
DENSITY_RANGE = (0.05, 25_000.0)


@dataclass
class Shape:
    shape_name: ClassVar[str] = "shape"
    mass: Any = None
    density: Any = None
    subtract: bool = False
    name: str | None = None

    def resolve(self, r: Resolver, what: str) -> mp.MassProperties:  # pragma: no cover
        raise NotImplementedError

    def _mass_args(self, r: Resolver, density_kind: str = "density") -> dict:
        density = None
        if self.density is not None:
            density = r.scalar(self.density, density_kind)
            if density_kind == "density" and not (DENSITY_RANGE[0] <= density <= DENSITY_RANGE[1]):
                raise InputError(
                    f"density {density:.4g} kg/m³ is not a physical solid or liquid "
                    f"({DENSITY_RANGE[0]:g} to {DENSITY_RANGE[1]:g} kg/m³). A bare number is "
                    f"read in {r.ctx.units.label('density')}; write the unit, e.g. "
                    "'7850 kg/m^3' or '7.85 g/cm^3'"
                )
        return {
            "mass": None if self.mass is None else r.scalar(self.mass, "mass"),
            "density": density,
        }

    @property
    def label(self) -> str:
        return self.name or self.shape_name


@dataclass
class PointMass(Shape):
    shape_name: ClassVar[str] = "point"
    at: Any = None

    def resolve(self, r, what):
        if self.density is not None:
            raise InputError("a point mass takes 'mass', not 'density'", what)
        if self.mass is None or self.at is None:
            raise InputError("a point mass needs 'mass' and 'at'", what)
        return mp.point_mass(
            r.scalar(self.mass, "mass"),
            r.position(self.at, f"{what}.at"),
            subtract=self.subtract,
            label=self.label,
        )


@dataclass
class Rod(Shape):
    """Slender rod; ``density`` here is mass per unit length."""

    shape_name: ClassVar[str] = "rod"
    start: Any = None
    end: Any = None

    def resolve(self, r, what):
        if self.start is None or self.end is None:
            raise InputError("a rod needs 'start' and 'end'", what)
        m = self._mass_args(r, "linear_density")
        return mp.rod(
            r.position(self.start, f"{what}.start"),
            r.position(self.end, f"{what}.end"),
            mass=m["mass"],
            linear_density=m["density"],
            subtract=self.subtract,
            label=self.label,
        )


@dataclass
class Box(Shape):
    shape_name: ClassVar[str] = "box"
    size: Any = None
    center: Any = None
    orientation: Any = None

    def resolve(self, r, what):
        if self.size is None or self.center is None:
            raise InputError("a box needs 'size' and 'center'", what)
        size = r.ctx.vector(self.size, "length", 3)
        return mp.box(
            size,
            r.position(self.center, f"{what}.center"),
            r.orientation(self.orientation, f"{what}.orientation"),
            subtract=self.subtract,
            label=self.label,
            **self._mass_args(r),
        )


@dataclass
class Cylinder(Shape):
    shape_name: ClassVar[str] = "cylinder"
    radius: Any = None
    length: Any = 0
    center: Any = None
    axis: Any = "+z"
    inner_radius: Any = 0

    def resolve(self, r, what):
        if self.radius is None or self.center is None:
            raise InputError("a cylinder needs 'radius' and 'center'", what)
        return mp.cylinder(
            r.scalar(self.radius, "length"),
            r.scalar(self.length, "length"),
            r.position(self.center, f"{what}.center"),
            r.axis(self.axis, f"{what}.axis"),
            inner_radius=r.scalar(self.inner_radius, "length"),
            subtract=self.subtract,
            label=self.label,
            **self._mass_args(r),
        )


@dataclass
class Sphere(Shape):
    shape_name: ClassVar[str] = "sphere"
    radius: Any = None
    center: Any = None
    inner_radius: Any = 0

    def resolve(self, r, what):
        if self.radius is None or self.center is None:
            raise InputError("a sphere needs 'radius' and 'center'", what)
        return mp.sphere(
            r.scalar(self.radius, "length"),
            r.position(self.center, f"{what}.center"),
            inner_radius=r.scalar(self.inner_radius, "length"),
            subtract=self.subtract,
            label=self.label,
            **self._mass_args(r),
        )


@dataclass
class Cone(Shape):
    shape_name: ClassVar[str] = "cone"
    radius: Any = None
    height: Any = None
    base_center: Any = None
    axis: Any = "+z"

    def resolve(self, r, what):
        if self.radius is None or self.height is None or self.base_center is None:
            raise InputError("a cone needs 'radius', 'height' and 'base_center'", what)
        return mp.cone(
            r.scalar(self.radius, "length"),
            r.scalar(self.height, "length"),
            r.position(self.base_center, f"{what}.base_center"),
            r.axis(self.axis, f"{what}.axis"),
            subtract=self.subtract,
            label=self.label,
            **self._mass_args(r),
        )


@dataclass
class CustomMass(Shape):
    """Mass properties from CAD or a datasheet.

    ``inertia`` is three diagonal values or a 3x3 tensor, in the axes given by
    ``orientation`` (default global), about ``about`` (default the cog).
    """

    shape_name: ClassVar[str] = "custom"
    cog: Any = None
    inertia: Any = None
    about: Any = None
    orientation: Any = None

    def resolve(self, r, what):
        if self.density is not None:
            raise InputError("custom mass properties take 'mass', not 'density'", what)
        if self.mass is None or self.cog is None:
            raise InputError("custom mass properties need 'mass' and 'cog'", what)
        inertia = None
        if self.inertia is not None:
            inertia = _inertia(r, self.inertia, f"{what}.inertia")
        props = mp.custom(
            r.scalar(self.mass, "mass"),
            r.position(self.cog, f"{what}.cog"),
            inertia,
            None if self.about is None else r.position(self.about, f"{what}.about"),
            r.orientation(self.orientation, f"{what}.orientation"),
            label=self.label,
        )
        if self.subtract:
            props = mp.MassProperties(-props.mass, props.cog, -props.inertia, props.label)
        return props


def _inertia(r: Resolver, value, what: str):
    import numpy as np

    if isinstance(value, list | tuple) and value and isinstance(value[0], list | tuple):
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise InputError("inertia matrix must be 3x3", what)
        return np.array([[r.scalar(v, "inertia") for v in row] for row in value])
    return r.ctx.vector(value, "inertia", 3)


SHAPES: dict[str, type[Shape]] = {
    cls.shape_name: cls for cls in (PointMass, Rod, Box, Cylinder, Sphere, Cone, CustomMass)
}
