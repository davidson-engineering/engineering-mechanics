import plotly.graph_objects as go
import numpy as np

from base.vector import BoundVector, Load, Reaction
from common.constants import PLOT_AXES


def _convert_unit_vector(vector: np.ndarray) -> np.ndarray:
    """
    Converts a vector to a unit vector. Handles the case where the vector norm is zero.

    Args:
        vector (np.ndarray): The vector to normalize.

    Returns:
        np.ndarray: The unit vector, or a zero vector if the input vector norm is zero.
    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros_like(vector)


def generate_hover_text(name, magnitude, units=None, labels=None) -> str:
    """
    Generates multiline custom hover text for a given vector's name and magnitude.

    Args:
        name (str): The name of the vector or moment.
        magnitude (np.ndarray): The magnitude values of the vector.
        units (list, optional): Units for each component. Defaults to ["N", "N", "N", "Nm", "Nm", "Nm"].
        labels (list, optional): Labels for each component. Defaults to ["Fx", "Fy", "Fz", "Mx", "My", "Mz"].

    Returns:
        str: A formatted string with hover text.
    """
    labels = labels or ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    units = units or ["N", "N", "N", "Nm", "Nm", "Nm"]
    if not magnitude[:3].any():
        # If all force components are zero, only include the moment components
        labels = labels[3:]
        magnitude = magnitude[3:]
    return f"{name}<br>" + "<br>".join(
        f"{label}: {value:.2f} {unit}"
        for label, value, unit in zip(labels, magnitude, units)
    )


def build_arrow(
    start, end, name="Arrow", color="blue", width=4, text="", arrow_size=0.4
):
    """
    Creates a 3D arrow with a line and a cone.

    Args:
        start (np.ndarray): Start point of the arrow.
        end (np.ndarray): End point of the arrow.
        name (str): Name of the arrow group.
        color (str): Color of the arrow.
        width (int): Width of the arrow line.
        text (str): Hover text for the arrow.
        arrow_size (float): Size reference for the arrowhead.

    Returns:
        list: A list of Plotly traces (line and cone).
    """
    start = np.asarray(start)
    end = np.asarray(end)
    vector = end - start

    # Line (vector body)
    body = go.Scatter3d(
        x=[start[0], end[0] * 0.99],
        y=[start[1], end[1] * 0.99],
        z=[start[2], end[2] * 0.99],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hoverinfo="text",
        text=text,
    )

    # Cone (vector tip)
    tip = go.Cone(
        x=[end[0]],
        y=[end[1]],
        z=[end[2]],
        u=[vector[0]],
        v=[vector[1]],
        w=[vector[2]],
        sizemode="absolute",
        sizeref=arrow_size,
        anchor="tip",
        colorscale=[[0, color], [1, color]],
        name=name,
        hoverinfo="text",
        text=text,
    )

    return [body, tip]


def build_load_traces(
    item, name=None, scale=1.0, as_components=False, labels=None, color="blue"
) -> list:
    """
    Creates Plotly traces for both force and moment vectors from a BoundVector object.

    Args:
        item (BoundVector): A vector object containing location and magnitude.
        name (str, optional): Name of the vector for display. Defaults to None.
        scale (float, optional): Scale factor for vector length. Defaults to 1.0.
        as_components (bool, optional): Whether to split vectors into components. Defaults to False.
        labels (list, optional): Custom labels for vector components. Defaults to None.
        color (str, optional): Color of the vector traces. Defaults to "blue".

    Returns:
        list: A list of Plotly traces.
    """
    if not isinstance(item, BoundVector):
        raise TypeError(f"Expected a BoundVector, got {type(item).__name__}")

    name = name or item.name or "Unnamed Vector"
    text = generate_hover_text(name, item.magnitude, labels)

    # Build force traces
    forces = build_force_traces(
        location=item.location,
        magnitude=item.magnitude[:3],
        name=name,
        scale=scale,
        as_components=as_components,
        text=text,
        color=color,
    )

    # Build moment traces
    moments = build_moment_traces(
        location=item.location,
        moments=item.magnitude[3:],
        name=name,
        scale=scale,
        text=text,
        color=color,
        as_components=as_components,
    )

    return forces + moments


def build_force_traces(
    location: np.ndarray,
    magnitude: np.ndarray,
    name: str = "Vector",
    scale: float = 1.0,
    color: str = "blue",
    width: int = 4,
    as_components: bool = False,
    component_colors=["red", "green", "blue"],
    text="",
    arrow_size=0.4,
):
    """
    Builds Plotly traces for 3D force vectors, optionally split into components.

    Args:
        location (np.ndarray): Starting point of the vector.
        magnitude (np.ndarray): Magnitude of the vector components.
        name (str): Name of the vector trace.
        scale (float): Scaling factor for vector length.
        color (str): Color for the vector body.
        width (int): Width of the vector line.
        as_components (bool): Whether to split the vector into XYZ components.
        component_colors (list): Colors for X, Y, Z components.
        text (str): Hover text for the vector.
        arrow_size (float): Size reference for arrowheads.

    Returns:
        list: Plotly traces for the vector.
    """
    location = np.asarray(location)
    magnitude = np.asarray(magnitude)

    if as_components:
        components = [
            (location, location + np.array([magnitude[0], 0, 0]) * scale),
            (location, location + np.array([0, magnitude[1], 0]) * scale),
            (location, location + np.array([0, 0, magnitude[2]]) * scale),
        ]
        traces = []
        for color, (start, end) in zip(component_colors, components):
            traces.extend(build_arrow(start, end, color=color, text=text, name=name))
        return traces

    return build_arrow(
        location,
        location + magnitude * scale,
        color=color,
        text=text,
        name=name,
        arrow_size=arrow_size,
    )


def build_moment_traces(
    location: np.ndarray,
    moments: np.ndarray,
    scale: float = 5,
    name: str = "Moments",
    color: str = "red",
    width: int = 4,
    as_components: bool = False,
    component_colors=["red", "green", "blue"],
    text="",
    base_radius: float = 2,
    start_offset: float = 0.3,
    end_offset: float = 1.1,
    resolution: int = 100,
    arrow_base_ref: float = 0.15,
    tolerance: float = 1e-3,
):
    """
    Builds Plotly traces for 3D rotational moments, optionally split into components.

    Args:
        location (np.ndarray): Center point of rotation.
        moments (np.ndarray): Moment magnitudes about X, Y, Z axes.
        scale (float): Scaling factor for arc size.
        name (str): Name of the trace group.
        color (str): Color of the arcs and cones.
        width (int): Line width for arcs.
        as_components (bool): Whether to split moments into X, Y, Z components.

    Returns:
        list: Plotly traces for rotational moments.
    """

    def build_curved_arrow(location, axis, text, color):
        theta = np.linspace(
            start_offset * 2 * np.pi, end_offset * 2 * np.pi, resolution
        )
        direction = np.array(axis["direction"])
        normal = np.array(axis["normal"])
        radius = base_radius * scale

        arc_points = (
            location
            + radius * np.outer(np.cos(theta), normal)
            + radius * np.outer(np.sin(theta), np.cross(direction, normal))
        )

        traces = [
            go.Scatter3d(
                x=arc_points[:-5, 0],
                y=arc_points[:-5, 1],
                z=arc_points[:-5, 2],
                mode="lines",
                line=dict(color=color, width=width),
                name=name,
                hoverinfo="text",
                text=text,
            )
        ]
        arrow_tip = arc_points[-1]
        tangent_start_idx = int(len(arc_points) * arrow_base_ref)
        tangent_vector = arc_points[-1] - arc_points[-tangent_start_idx]
        traces.append(
            go.Cone(
                x=[arrow_tip[0]],
                y=[arrow_tip[1]],
                z=[arrow_tip[2]],
                u=[tangent_vector[0]],
                v=[tangent_vector[1]],
                w=[tangent_vector[2]],
                sizemode="absolute",
                sizeref=radius * 0.8,
                colorscale=[[0, color], [1, color]],
                anchor="tip",
                name=f"{name} Tip",
                hoverinfo="text",
                text=text,
            )
        )
        return traces

    traces = []
    location = np.asarray(location)
    moments = np.asarray(moments)

    if as_components:
        for (i, axis), color in zip(enumerate(PLOT_AXES), component_colors):
            if moments[i] >= tolerance:
                traces += build_curved_arrow(location, axis, text, color)
    else:
        if np.linalg.norm(moments) >= tolerance:
            axis = {
                "direction": _convert_unit_vector(moments),
                "normal": _convert_unit_vector(np.cross(moments, [0, 0, 1])),
            }
            traces += build_curved_arrow(location, axis, text, color)

    return traces
