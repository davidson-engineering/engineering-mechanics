# engineering_mechanics/visualization/plot3d.py

import plotly.graph_objects as go
import numpy as np

from base.assembly import Assembly
from base.plotting import (
    build_load_traces,
    build_part_traces,
    build_coord_sys_traces,
    compute_bounds_from_traces,
)
from statics.study import AssemblyStudy


def plot_loads_3d(loads=None, parts=None, scale=1.0):
    """
    Visualizes resulting loads in 3D using Plotly.

    Args:
        forces (np.ndarray): An Nx3 array of force vectors (Fx, Fy, Fz) at each node.
        moments (np.ndarray): An Nx3 array of moment vectors (Mx, My, Mz) at each node.
        nodes (np.ndarray): An Nx3 array of node positions (x, y, z).
        scale (float): Scale factor for visualizing the force and moment vectors.

    Example:
        forces = np.array([[100, 0, 0], [0, 200, 0]])
        moments = np.array([[0, 0, 50], [0, 0, -50]])
        nodes = np.array([[0, 0, 0], [1, 1, 0]])
        plot_loads_3d(forces, moments, nodes)
    """

    fig = go.Figure()

    traces = []

    # Show coordinate system
    traces.extend(build_coord_sys_traces())

    # if loads:
    #     for load in loads:
    #         traces.extend(build_load_traces(load, scale=scale, as_components=False))
    if parts:
        traces.extend(
            build_part_traces(parts, mass_scale=0.2, reaction_scale=1, load_scale=1)
        )

    # Add nodes as scatter points
    [fig.add_trace(trace) for trace in traces]

    x_bounds, y_bounds, z_bounds = compute_bounds_from_traces(traces, padding=1)

    # Set 3D layout
    # lock aspect ratio to 1:1:1
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_bounds),
            yaxis=dict(range=y_bounds),
            zaxis=dict(range=z_bounds),
            aspectmode="data",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        title="3D Load Visualization",
    )
    # Disable colorbar for all traces
    for trace in fig.data:
        if hasattr(
            trace, "showscale"
        ):  # Check if the trace has a 'showscale' attribute
            trace.showscale = False

    # fig.update_traces(lighting=dict(ambient=0.5, specular=1.0))

    fig.show()


def assembly_plot_test():

    from base.vector import Reaction, Load
    from base.assembly import Part, Connection

    from mechanics import Rod, Disc

    rod1 = Rod(id="rod", length=1, mass=10, cog=[1, 0, 0])
    rod_connect_R = Reaction(name=1, location=[1, 0, 0], constraint=np.eye(6))
    rod_ground = Reaction(name=0, location=[0, 0, 0], constraint=np.eye(6))
    ground = Reaction(name="ground", location=[0, 0, 0], constraint=np.eye(6))

    disc1 = Disc(id="disc", radius=0.5, mass=100, cog=[0, 0, 0])
    disc_connect_R = Reaction(name=2, location=[1, 0, 0], constraint=np.eye(6))
    rod_disc_connection = Connection(master=rod_connect_R, slave=disc_connect_R)
    rod_ground_connection = Connection(master=rod_ground, slave=ground)
    loads = [
        Load(name="disc_load", magnitude=[10, 10, 10, 0, 0, 0], location=[2, 0, 0])
    ]

    rod_part = Part(
        id="rod_part",
        bodies=[rod1],
        connections=[rod_disc_connection, rod_ground_connection],
    )
    disc_part = Part(
        id="disc_part",
        bodies=[disc1],
        connections=[rod_disc_connection.invert()],
        loads=loads,
    )
    parts = [rod_part, disc_part]

    study = AssemblyStudy(
        name="Test Study",
        description="Test study with one load and one reaction",
        assembly=Assembly(parts=parts),
    )
    result = study.run()

    plot_loads_3d(loads=result.solution.loads, parts=result.solution.parts, scale=1.0)


def simple_test():
    from base.vector import Load

    load1 = Load(name="load1", magnitude=[10, 10, 10, 10, 10, -10], location=[1, 1, 0])
    plot_loads_3d(loads=[load1])


if __name__ == "__main__":

    # simple_test()
    assembly_plot_test()
