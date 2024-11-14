#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2024-11-09
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""Module for modeling collections of bodies to solve engineering mechanics problems."""
# ---------------------------------------------------------------------------

import numpy as np
from scipy.spatial.transform import Rotation as R
from common.types import BoundVector, Load
from common.constants import GRAVITY
import logging


logger = logging.getLogger(__name__)


def skew_sym(x) -> np.array:
    """
    Return the skew-symmetric matrix of a 3D vector.

    Args:
        x (array-like): A 3-element array representing a vector.

    Returns:
        np.array: 3x3 skew-symmetric matrix.
    """
    if isinstance(x, list):
        x = np.array(x)
    if x.shape == (3,):
        a, b, c = x
    else:
        return None
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])


def rot_vec(angle, axis):
    """
    Calculate a rotation vector by scaling the rotation axis with the angle.

    Args:
        angle (float): The rotation angle in radians.
        axis (array-like): A 3-element array representing the rotation axis.

    Returns:
        np.array: Rotation vector.
    """
    return angle * np.array(axis)


class Body:
    """
    A class representing a rigid body in 3D space.
    """

    no_bodies = 1

    def __init__(
        self, id=None, mass=1.0, cog=np.zeros(3), inertia=np.zeros(3), rot=None
    ) -> None:
        """
        Initialize a Body object.

        Args:
            id (str, optional): Identifier for the body.
            mass (float): Mass of the body.
            cog (array-like): Center of gravity of the body in 3D space.
            inertia (array-like): Inertia tensor (diagonal elements or full matrix).
            rot (array-like, optional): Initial rotation vector for orientation.
        """
        self.id = Body.no_bodies if id is None else id
        self.mass = mass
        self.cog = np.asarray(cog)
        self.inertia = (
            np.diag(inertia)
            if (np.array(inertia).shape == (1, 3)) or (np.array(inertia).shape == (3,))
            else np.asarray(inertia)
        )

        if rot is not None:
            rotation = R.from_rotvec(rot).as_matrix()
            self.inertia = rotation @ self.inertia @ rotation.T

        Body.no_bodies += 1

    def get_inertia(self, ref=None) -> np.array:
        """
        Calculate the inertia tensor relative to a reference point.

        Args:
            ref (array-like, optional): Reference point for inertia calculation. Defaults to the center of gravity.

        Returns:
            np.array: Inertia tensor relative to the reference point.
        """
        if ref is None:
            ref = self.cog
        elif str(ref) == "0":
            ref = np.zeros(3)

        return self.inertia - self.mass * np.linalg.matrix_power(
            skew_sym(np.asarray(ref) - self.cog), 2
        )

    def get_prin_inertia(self):
        """
        Calculate the principal moments of inertia.

        Returns:
            np.array: Principal moments of inertia.
        """
        return np.diag(np.linalg.eig(self.inertia)[0])

    def get_torque(self, vel=np.zeros(3), accel=np.zeros(3), ref=None) -> np.array:
        """
        Calculate the torque required for a given acceleration.

        Args:
            vel (array-like): Angular velocity of the body.
            accel (array-like): Angular acceleration of the body.
            ref (array-like, optional): Reference point for the torque calculation.

        Returns:
            np.array: Torque vector.
        """
        if ref is None:
            ref = self.cog
        vel = np.asarray(vel)
        accel = np.asarray(accel)
        return self.get_inertia(ref) @ accel + np.cross(
            vel, self.get_inertia(ref) @ accel
        )

    def _get_weight(self, gravity=GRAVITY) -> np.array:
        """
        Calculate the weight of the body.

        Args:
            gravity (array-like): Gravitational acceleration vector.

        Returns:
            np.array: Weight vector.
        """
        assert gravity.shape == (3,) or gravity.shape == (
            6,
        ), "Gravity vector must be 3D or 6D"
        # pad gravity vector to 6D
        gravity = np.pad(gravity, (0, 6 - gravity.size))
        return self * gravity

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(
                mass=self.mass * other, cog=self.cog, inertia=self.inertia
            )
        if isinstance(other, (list, tuple)):
            other = np.asarray(other)
        if isinstance(other, (BoundVector, np.ndarray)):
            return Load(magnitude=self.mass * other, location=self.cog)

    def __add__(self, other):
        if isinstance(other, Body):
            return Bodies(bodies=[self, other])
        if isinstance(other, Bodies):
            other.add_body(self)
            return other


class Bodies(Body):
    """
    A class representing a collection of bodies.
    """

    def __init__(self, bodies=None, *args, **kwargs) -> None:
        """
        Initialize a Bodies collection.

        Args:
            bodies (list): List of Body objects.
        """
        super().__init__(*args, **kwargs)
        self.bodies = [] if bodies is None else bodies
        self.update_bodies()

    def update_bodies(self, bodies=None) -> None:
        """
        Update the collective properties (mass, center of gravity, inertia) of the collection.

        Args:
            bodies (list, optional): New list of Body objects to include in the collection.
        """
        if bodies is not None:
            self.bodies = bodies
        self.mass = np.sum([body.mass for body in self.bodies])
        self.cog = (
            np.sum([body.mass * body.cog for body in self.bodies], axis=0) / self.mass
            if bool(self.mass)
            else 0
        )
        self.inertia = np.sum(
            [body.get_inertia(ref=self.cog) for body in self.bodies], axis=0
        )

    def remove_body(self, body) -> None:
        """
        Remove a body from the collection.

        Args:
            body (Body): The Body object to remove.
        """
        try:
            self.bodies.remove(body)
            self.update_bodies()
        except ValueError as e:
            logger.error(e.__class__, "has occurred: Body not in bodies list")

    def add_body(self, body) -> None:
        """
        Add a body to the collection.

        Args:
            body (Body): The Body object to add.
        """
        self.bodies.append(body)
        self.update_bodies()

    def __add__(self, other):
        return self.add_body(other)

    def __sub__(self, other):
        return self.remove_body(other)


class Rod(Body):
    """
    A class representing a rod-shaped body.
    """

    def __init__(self, mass=1.0, length=1.0, *args, **kwargs) -> None:
        """
        Initialize a Rod object.

        Args:
            mass (float): Mass of the rod.
            length (float): Length of the rod.
        """
        inertia = mass * length**2 / 12 * np.diag([1, 1, 0])
        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)
        self.length = length


class Disc(Body):
    """
    A class representing a disc-shaped body.
    """

    def __init__(self, mass=1.0, radius=1.0, *args, **kwargs) -> None:
        """
        Initialize a Disc object.

        Args:
            mass (float): Mass of the disc.
            radius (float): Radius of the disc.
        """
        inertia = mass * radius**2 / 4 * np.diag([1, 1, 2])
        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)
        self.radius = radius


class Cuboid(Body):
    """
    A class representing a cuboid-shaped body.
    """

    def __init__(self, mass=1.0, width=1.0, depth=1.0, height=1.0, *args, **kwargs):
        """
        Initialize a Cuboid object.

        Args:
            mass (float): Mass of the cuboid.
            width (float): Width of the cuboid.
            depth (float): Depth of the cuboid.
            height (float): Height of the cuboid.
        """
        inertia = (
            mass
            / 12
            * np.diag([depth**2 + height**2, width**2 + height**2, width**2 + depth**2])
        )
        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)
        self.width = width
        self.depth = depth
        self.height = height


class Cylinder(Body):
    """
    A class representing a cylinder-shaped body.
    """

    def __init__(
        self, mass=1.0, oradius=1.0, iradius=None, height=1.0, *args, **kwargs
    ):
        """
        Initialize a Cylinder object.

        Args:
            mass (float): Mass of the cylinder.
            oradius (float): Outer radius of the cylinder.
            iradius (float, optional): Inner radius of the cylinder. Defaults to oradius.
            height (float): Height of the cylinder.
        """
        if iradius is None:
            iradius = oradius

        inertia = (
            mass
            / 12
            * np.diag(
                [
                    3 * (oradius**2 + iradius**2) + height**2,
                    3 * (oradius**2 + iradius**2) + height**2,
                    6 * (oradius**2 + iradius**2),
                ]
            )
        )
        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)
        self.oradius = oradius
        self.iradius = iradius
        self.height = height


if __name__ == "__main__":

    info = (
        lambda x, y="inertia": "\n"
        + x.id
        + " with mass "
        + str(x.mass)
        + "kg has "
        + str(y)
        + ":"
    )

    rod = Rod(cog=[0, 0, 0], mass=2.0, id="rod")

    print(info(rod))

    print(i := rod.get_inertia())

    disc = Disc(cog=[0, 0, 0], mass=1.2, radius=1.8, id="disc")

    print(info(disc))

    print(i := disc.get_inertia())

    # combine rod and disc to form spindle

    spindle = Bodies(bodies=[rod, disc], id="spindle")

    print(info(spindle))

    print(i := spindle.get_inertia())

    print(info(spindle, "inertia about principal axes"))

    print(i := spindle.get_prin_inertia())

    print(info(spindle, "torque about cog"))

    print(T := spindle.get_torque(accel=[0, 0, 2], vel=[0, 0, 2]))

    print(info(spindle, "torque about remote point [1,0,1]"))

    print(T := spindle.get_torque(accel=[0, 0, 2], vel=[0, 0, 2], ref=[1, 0, 1]))

    # create rod with length in x, rotate object by 90 degrees in z so length is along y

    rot = rot_vec(np.pi / 2, [0, 0, 1])

    rod = Rod(length=1.0, cog=[0, 0, 0], mass=1.0, rot=rot, id="rod")

    print(f"\n{rod.id} has inertia about cog :\n", i := rod.get_inertia())

    print(
        f"\n{rod.id} has inertia about one end:\n",
        i := rod.get_inertia(ref=[0, rod.length / 2, 0]),
    )

    # rotate spindle by 45 degrees about z-axis

    spindle = Bodies(
        bodies=[rod, disc], id="rotated spindle", rot=rot_vec(np.pi / 4, [0, 0, 1])
    )

    print(info(spindle))

    print(i := spindle.get_inertia())

    print(info(spindle, f"inertia about remote point [1,2,3]"))

    print(i := spindle.get_inertia(ref=[1, 2, 3]))

    gripper = Cylinder(
        id="gripper",
        mass=0.804,
        iradius=0.0692 / 2,
        oradius=0.0381,
        height=0.129,
        cog=[0.00191, 0, 0.05080],
    )

    obj = Cuboid(
        id="object",
        mass=0.064,
        width=0.0762,
        depth=0.0254,
        height=0.00425,
        cog=[0.0127, 0, 0.128],
    )

    print(info(gripper))

    print(gripper.get_inertia())

    print(" ")

    print(gripper.get_inertia(ref=0))

    print(info(obj))

    print(obj.get_inertia())

    print(" ")

    print(obj.get_inertia(ref="0"))

    payload = Bodies(id="payload", bodies=[gripper, obj])

    print(info(payload))

    print(payload.get_inertia())

    print(" ")

    print(payload.get_inertia(ref="0"))
