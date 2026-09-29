"""The input file format, as pydantic models.

The schema checks structure: known fields, required fields, and the ``type``
of each support, joint and shape. Values that carry units (``"10 kN"``,
``[0, -5] kN``, ``L/2``, a point name ...) are kept as raw input here and
interpreted by :mod:`engmech.inputs`, which knows the unit system,
parameters and named points.

``engmech schema`` exports this as JSON Schema for editor autocompletion.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Raw values: numbers or strings with units/expressions, or lists of them.
Scalar = Annotated[
    Any, Field(json_schema_extra={"anyOf": [{"type": "number"}, {"type": "string"}]})
]
Vector = Annotated[
    Any,
    Field(
        json_schema_extra={
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}},
            ]
        }
    ),
]
Position = Annotated[
    Any,
    Field(
        description="A named point, or coordinates like [x, y, z] (planar: [x, y])",
        json_schema_extra={
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}},
            ]
        },
    ),
]
Direction = Annotated[
    Any,
    Field(
        description="'+x', '-y', 'z', a vector, or (planar) an angle like '30 deg'",
        json_schema_extra={
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}},
            ]
        },
    ),
]
Flexible = Annotated[Any, Field(json_schema_extra={})]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- shapes


class _Shape(Strict):
    name: str | None = None
    mass: Scalar = None
    density: Scalar = Field(None, description="mass per volume (rod: mass per length)")
    subtract: bool = Field(False, description="remove this shape (a hole or cut-out)")


class PointShape(_Shape):
    type: Literal["point"]
    at: Position


class RodShape(_Shape):
    type: Literal["rod"]
    start: Position
    end: Position


class BoxShape(_Shape):
    type: Literal["box"]
    size: Vector = Field(description="edge lengths along the local x, y, z axes")
    center: Position
    orientation: Flexible = None


class CylinderShape(_Shape):
    type: Literal["cylinder", "disc"]
    radius: Scalar
    length: Scalar = Field(0, description="axial length; 0 for a thin disc")
    center: Position
    axis: Direction = "+z"
    inner_radius: Scalar = 0


class SphereShape(_Shape):
    type: Literal["sphere"]
    radius: Scalar
    center: Position
    inner_radius: Scalar = 0


class ConeShape(_Shape):
    type: Literal["cone"]
    radius: Scalar
    height: Scalar
    base_center: Position
    axis: Direction = Field("+z", description="from the base towards the apex")


class CustomShape(_Shape):
    type: Literal["custom"]
    cog: Position
    inertia: Flexible = Field(None, description="[Ixx, Iyy, Izz] or a 3x3 tensor")
    about: Position | None = Field(
        None, description="point the inertia is given about (default cog)"
    )
    orientation: Flexible = None


ShapeSpec = Annotated[
    PointShape | RodShape | BoxShape | CylinderShape | SphereShape | ConeShape | CustomShape,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- loads


class Distributed(Strict):
    start: Position
    end: Position
    intensity: Flexible = Field(description="force per length, or {start: w1, end: w2}")
    direction: Direction
    projected: bool = Field(
        False, description="intensity per unit projected length (e.g. snow on a slope)"
    )


class LoadSpec(Strict):
    """One load: exactly one of force, moment, distributed or unknown."""

    name: str | None = None
    body: str | None = None
    case: str | None = None
    force: Flexible = Field(
        None, description="vector, or {magnitude, direction | angle | toward | along}"
    )
    moment: Flexible = Field(None, description="vector, planar scalar, or {magnitude, axis}")
    distributed: Distributed | None = None
    unknown: str | None = Field(
        None, description="solve for this load's magnitude; give 'direction' or 'axis'"
    )
    at: Position | None = None
    direction: Direction | None = None
    axis: Direction | None = None


class Motion(Strict):
    angular_velocity: Flexible = None
    angular_acceleration: Flexible = None
    acceleration: Vector | None = Field(None, description="acceleration of the centre of gravity")
    pivot: Position | None = Field(None, description="a point with known acceleration")
    pivot_acceleration: Vector | None = None
    case: str | None = None


class BodySpec(Strict):
    mass: Scalar = None
    cog: Position | None = None
    inertia: Flexible = None
    shapes: list[ShapeSpec] = Field(default_factory=list)
    particle: bool = Field(False, description="only forces act through a single point")
    motion: Motion | None = None
    outline: list[Position] | None = Field(None, description="points to draw the body through")
    loads: list[LoadSpec] = Field(default_factory=list)


# --------------------------------------------------------------------------- joints


class _Joint(Strict):
    body: str | None = Field(None, description="the supported body (supports only)")
    bodies: list[str] | None = Field(None, description="[a, b] (joints only)")


class FixedJoint(_Joint):
    type: Literal["fixed", "weld"]
    at: Position
    orientation: Flexible = None
    stiffness: Flexible = None


class PinJoint(_Joint):
    type: Literal["pin", "revolute", "hinge"]
    at: Position
    axis: Direction | None = Field(None, description="rotation axis (planar: z)")
    actuated: bool = Field(False, description="solve for the drive torque about the axis")
    stiffness: Flexible = None


class BallJoint(_Joint):
    type: Literal["ball", "spherical"]
    at: Position
    stiffness: Flexible = None


class SliderJoint(_Joint):
    type: Literal["slider", "prismatic"]
    at: Position
    axis: Direction
    actuated: bool = Field(False, description="solve for the drive force along the axis")
    stiffness: Flexible = None


class CylindricalJoint(_Joint):
    type: Literal["cylindrical"]
    at: Position
    axis: Direction
    stiffness: Flexible = None


class BearingJoint(_Joint):
    type: Literal["bearing"]
    at: Position
    axis: Direction
    thrust: bool = Field(False, description="also carry force along the axis")
    stiffness: Flexible = None


class UniversalJoint(_Joint):
    type: Literal["universal"]
    at: Position
    axes: list[Direction]
    stiffness: Flexible = None


class RollerJoint(_Joint):
    type: Literal["roller", "contact"]
    at: Position
    normal: Direction = Field(description="direction the support pushes on the body")
    stiffness: Scalar = None


class LinkJoint(_Joint):
    type: Literal["link", "strut", "cable"]
    at: Position | None = Field(None, description="support: attachment point on the body")
    anchor: Position | None = Field(None, description="support: the fixed end")
    ends: list[Position] | None = Field(None, description="joint: [point on a, point on b]")
    stiffness: Scalar = None


class CustomJoint(_Joint):
    type: Literal["custom"]
    at: Position
    constrain: list[Literal["Fx", "Fy", "Fz", "Mx", "My", "Mz"]]
    orientation: Flexible = None
    stiffness: Flexible = None


JointSpec = Annotated[
    FixedJoint
    | PinJoint
    | BallJoint
    | SliderJoint
    | CylindricalJoint
    | BearingJoint
    | UniversalJoint
    | RollerJoint
    | LinkJoint
    | CustomJoint,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- top level


class CheckSpec(Strict):
    target: str = Field(description="'<joint>.<component>' or '<body>.mass'")
    name: str | None = None
    case: str | None = None
    expect: Scalar = None
    tolerance: Scalar = Field("0.1%", description="absolute, or a percentage like '1%'")
    max: Scalar = None
    min: Scalar = None


class ModelFile(Strict):
    """An engmech model file."""

    model_config = ConfigDict(extra="forbid", title="engmech model")

    name: str = "Untitled model"
    description: str = ""
    analysis: Literal["spatial", "planar"] = "spatial"
    units: Flexible = Field(None, description="preset name or {length, force, mass, angle, time}")
    output_units: Flexible = None
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    gravity: Flexible = Field(None, description="'-z', a vector, or {direction, magnitude, case}")
    points: dict[str, Position] = Field(default_factory=dict)
    bodies: dict[str, BodySpec | None] = Field(default_factory=dict)
    supports: dict[str, JointSpec] = Field(default_factory=dict)
    joints: dict[str, JointSpec] = Field(default_factory=dict)
    loads: list[LoadSpec] = Field(default_factory=list)
    combinations: dict[str, dict[str, Scalar]] = Field(default_factory=dict)
    checks: list[CheckSpec] = Field(default_factory=list)


def json_schema() -> dict:
    schema = ModelFile.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema
