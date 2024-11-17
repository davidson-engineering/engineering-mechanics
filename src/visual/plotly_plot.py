# engineering_mechanics/visualization/plot3d.py

import plotly.graph_objects as go
import numpy as np

from base.plotting import build_load_traces


def plot_loads_3d(vectors, scale=1.0):
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
    for i, vector in enumerate(vectors):
        traces.extend(
            build_load_traces(
                vector, name=f"Load {i+1}", scale=scale, as_components=False
            )
        )

    # Add nodes as scatter points
    [fig.add_trace(trace) for trace in traces]

    # Set 3D layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        title="3D Load Visualization",
    )

    fig.show()


if __name__ == "__main__":

    from base.vector import Reaction

    # Example usage
    loads = [
        Reaction(
            location=np.array([5, 5, 5]),
            magnitude=np.array([5, 10, 15, 0.0005, 1, 0]),
        ),
        Reaction(
            location=np.array([0, 0, 0]),
            magnitude=np.array([10, 10, 20, 50, 100, 10]),
        ),
    ]

    plot_loads_3d(loads, scale=0.2)
