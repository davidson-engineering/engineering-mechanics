#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Matthew Davidson
# Created Date: 2024-01-01
# version ='0.0.1'
# ---------------------------------------------------------------------------
"""a_short_module_description"""
# ---------------------------------------------------------------------------

# Mechanics Module
# Author : Matthew Davidson
# Module to model collections of bodies for solving engineering problems

import numpy as np
from scipy.spatial.transform import Rotation as R

def skewSym(X) -> np.array:
    # Return skew symmetric matrix of input X
    if isinstance(X, list): X = np.array(X)
    if X.shape == (3,): a, b, c = X
    else: return None
    return np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])

def rotVec(ang, axis):
    return ang * np.array(axis)

class Body:

    noBodies = 1

    def __init__(self, id=None, mass=1.0, cog=np.zeros(3), inertia=np.zeros(3), rot=None) -> None:
        self.id = Body.noBodies if id is None else id
        self.mass = mass
        self.cog = np.asarray(cog)
        self.inertia = np.diag(inertia) if np.array(
            inertia).shape == (1, 3) else np.asarray(inertia)
        if rot is not None:
            Rot = R.from_rotvec(rot).as_matrix()
            self.inertia = Rot@self.inertia@Rot.T
        Body.noBodies += 1

    def getInertia(self, ref=None) -> np.array:

        if ref is None:
            ref = self.cog

        elif str(ref) == '0':
            ref = np.zeros(3)

        return self.inertia - self.mass*np.linalg.matrix_power(skewSym(np.asarray(ref)-self.cog), 2)

    def getPrinInertia(self):

        return np.diag(np.linalg.eig(self.inertia)[0])

    def getTorque(self, vel=np.zeros(3), accel=np.zeros(3), ref=None) -> None:

        if ref is None: ref = self.cog
        vel = np.asarray(vel)
        accel = np.asarray(accel)
        return self.getInertia(ref)@accel + np.cross(vel, self.getInertia(ref)@accel)

class Bodies(Body):

    def __init__(self, bodies=[], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bodies = bodies
        self.updateBodies()

    def updateBodies(self, bodies=None) -> None:

        if bodies is not None: self.bodies = bodies
        self.mass = np.sum([body.mass for body in self.bodies])
        self.cog = np.sum([body.mass*body.cog for body in self.bodies],
            axis=0)/self.mass if bool(self.mass) else 0

        self.inertia = np.sum([body.getInertia(ref=self.cog) for body in self.bodies], axis=0)

    def removeBody(self, body) -> None:

        try:
            self.bodies.remove(body)
            self.updateBodies()

        except ValueError as e:
            print(e.__class__, "has occured : Body not in bodies list")

    def addBody(self, body) -> None:
        self.bodies.append(body)
        self.updateBodies()

class Rod(Body):

    def __init__(self, mass=1.0, length=1.0, *args, **kwargs) -> None:
        inertia = mass*length**2/12 * np.diag([1, 1, 0])
        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)
        self.length = length

class Disc(Body):

    def __init__(self, mass=1.0, radius=1.0, *args, **kwargs) -> None:
        inertia = mass*radius**2/4 * np.diag([1, 1, 2])
        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)
        self.radius = radius

class Cuboid(Body):

    def __init__(self, mass=1.0, width=1.0, depth=1.0, height=1.0, *args, **kwargs):
        
        inertia = mass/12 * \
            np.diag([depth**2+height**2, width**2+height**2, width**2+depth**2])
        
        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)
        
        self.width = width
        self.depth = depth
        self.height = height

class Cylinder(Body):

    def __init__(self, mass=1.0, oradius=1.0, iradius=None, height=1.0, *args, **kwargs):

        if iradius is None: iradius = oradius

        inertia = mass/12 * np.diag(
            [3*(oradius**2+iradius**2)+height**2,
             3*(oradius**2+iradius**2)+height**2,
             6*(oradius**2+iradius**2)])

        super().__init__(inertia=inertia, mass=mass, *args, **kwargs)

        self.oradius = oradius
        self.iradius = oradius
        self.height = height

if __name__ == "__main__":

    info = lambda x, y="inertia": "\n" + x.id + \
        " with mass " + str(x.mass) + "kg has " + str(y) + ":"

    rod = Rod(cog=[0, 0, 0], mass=2.0, id='rod')

    print(info(rod))

    print(i := rod.getInertia())

    disc = Disc(cog=[0, 0, 0], mass=1.2, radius=1.8, id='disc')

    print(info(disc))

    print(i := disc.getInertia())

    # combine rod and disc to form spindle

    spindle = Bodies(bodies=[rod, disc], id='spindle')

    print(info(spindle))

    print(i := spindle.getInertia())

    print(info(spindle, "inertia about principal axes"))

    print(i := spindle.getPrinInertia())

    print(info(spindle, "torque about cog"))

    print(T := spindle.getTorque(accel=[0, 0, 2], vel=[0, 0, 2]))

    print(info(spindle, "torque about remote point [1,0,1]"))

    print(T := spindle.getTorque(
        accel=[0, 0, 2], vel=[0, 0, 2], ref=[1, 0, 1]))

    # create rod with length in x, rotate object by 90 degrees in z so length is along y

    rot = rotVec(np.pi/2, [0, 0, 1])

    rod = Rod(length=1.0, cog=[0, 0, 0], mass=1.0, rot=rot, id='rod')

    print(f"\n{rod.id} has inertia about cog :\n", i := rod.getInertia())

    print(f"\n{rod.id} has inertia about one end:\n",
          i := rod.getInertia(ref=[0, rod.length/2, 0]))

    # rotate spindle by 45 degrees about z-axis

    spindle = Bodies(
        bodies=[rod, disc], id='rotated spindle', rot=rotVec(np.pi/4, [0, 0, 1]))

    print(info(spindle))

    print(i := spindle.getInertia())

    print(info(spindle, f"inertia about remote point [1,2,3]"))

    print(i := spindle.getInertia(ref=[1, 2, 3]))

    gripper = Cylinder(id='gripper', mass=0.804, iradius=0.0692/2,
                       oradius=0.0381, height=0.129, cog=[0.00191, 0, 0.05080])

    object = Cuboid(id='object', mass=0.064, width=0.0762,
                    depth=0.0254, height=0.00425, cog=[0.0127, 0, 0.128])

    print(info(gripper))

    print(gripper.getInertia())

    print(' ')

    print(gripper.getInertia(ref=0))

    print(info(object))

    print(object.getInertia())

    print(' ')

    print(object.getInertia(ref='0'))

    payload = Bodies(id='payload', bodies=[gripper, object])

    print(info(payload))

    print(payload.getInertia())

    print(' ')

    print(payload.getInertia(ref='0'))
