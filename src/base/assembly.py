from typing import List, Union

import numpy as np
from mechanics.mechanics import Bodies, Body
from base.vector import Load, Reaction
from scipy.spatial import ConvexHull


class Connection(list):
    def __init__(self, master: Reaction, slave: Reaction, id=None):
        super().__init__()
        self.master = master
        self.slave = slave
        self.id = id

    def invert(self):
        return self.__class__(id=self.id, master=self.slave, slave=self.master)

    def __repr__(self):
        return f"Connection(id={self.id}, master={self.master.name}, slave={self.slave.name})"


class Part(Bodies):
    def __init__(
        self,
        id,
        bodies: Union[Bodies, Body],
        connections: List[Connection] = None,
        loads: List[Load] = None,
    ):
        super().__init__(id=id, bodies=bodies)
        self.connections = [] if connections is None else connections
        self.loads = [] if loads is None else loads
        self.reactions = [connection.master for connection in self.connections]
        self.nodes = [reaction.location for reaction in self.reactions]
        # self.faces = self.generate_faces()

    def __repr__(self):
        return f"Part(id={self.id}, bodies={len(self.bodies)}, connections={len(self.connections)}, loads={len(self.loads)})"

    def updates_nodes(self, nodes: List[np.ndarray] = None):
        self.nodes = (
            [reaction.location for reaction in self.reactions]
            if nodes is None
            else nodes
        )

    def generate_faces(self):
        """
        Automatically generates faces using the ConvexHull algorithm.

        Returns:
            List[List[int]]: A list of triangular faces, where each face is defined by three node indices.
        """
        nodes_array = np.array(self.nodes)
        hull = ConvexHull(nodes_array)
        return hull.simplices


class Assembly:
    def __init__(self, parts: List[Part]):
        self.parts = parts
        self.loads = [load for part in self.parts for load in part.loads]

    def construct_terms(self):
        pass

    def solve(self):
        pass


if __name__ == "__main__":
    from mechanics import Rod, Disc

    rod1 = Rod(id="rod", length=1, mass=1)
    rod_connect_R = Reaction(name=1, location=[1, 0, 0], constraint=np.eye(6))
    rod_ground = Reaction(name=0, location=[0, 0, 0], constraint=np.eye(6))
    ground = Reaction(name="ground", location=[0, 0, 0], constraint=np.eye(6))

    disc1 = Disc(id="disc", radius=0.5, mass=1)
    disc_connect_R = Reaction(name=2, location=[1, 0, 0], constraint=np.eye(6))
    rod_disc_connection = Connection(master=rod_connect_R, slave=disc_connect_R)
    rod_ground_connection = Connection(master=rod_ground, slave=ground)
    load = Load(name="disc_load", magnitude=[-10, 0, -10, 0, 0, 0], location=[2, 0, 0])

    rod_part = Part(
        id="rod_part",
        bodies=[rod1],
        connections=[rod_disc_connection, rod_ground_connection],
    )
    disc_part = Part(
        id="disc_part",
        bodies=[disc1],
        connections=[rod_disc_connection.invert()],
        loads=[load],
    )

    assembly = Assembly(parts=[rod_part, disc_part])

    print(assembly.parts)
