from enum import Enum
import itertools
from typing import List
import plotly.graph_objects as go
import numpy as np


from base.assembly import Part
from base.vector import BoundVector, Load, Reaction
from common.constants import PLOT_AXES


# Define arrow type enum
class ArrowType(Enum):
    DOUBLE_ARROW = 0
    SINGLE_ARROW = 1
    CURVED_ARROW = 2


def ensure_minimum_vector(
    vector: np.ndarray, scale: float, threshold: float = 1.0
) -> np.ndarray:
    """
    Ensures that the vector magnitude is above a specified threshold.

    Args:
        vector (np.ndarray): The input vector to scale and check.
        scale (float): The scale factor for the vector.
        threshold (float): The minimum allowed magnitude for the vector.

    Returns:
        np.ndarray: The scaled vector with a magnitude above the threshold.
    """
    vector = np.asarray(vector) * scale
    magnitude = np.linalg.norm(vector)

    if magnitude < threshold:
        if magnitude > 0:
            vector = (vector / magnitude) * threshold  # Scale to the threshold
        else:
            vector = np.array(
                [threshold, 0, 0]
            )  # Default to x-direction for zero vectors

    return vector


def build_coord_sys_traces(
    scale: float = 1.0,
    colors: tuple[str] = ("red", "green", "blue"),
    width: int = 6,
    location=(0, 0, 0),
    opacity=0.3,
) -> List[go.Scatter3d]:
    """
    Builds Plotly traces for a 3D coordinate system.

    Args:
        scale (float): Scaling factor for the coordinate system.

    Returns:
        list: A list of Plotly traces for the coordinate system.
    """
    traces = []
    for axis, color in zip(PLOT_AXES, colors):
        traces.append(
            go.Scatter3d(
                x=[location[0], axis["direction"][0] * scale],
                y=[location[1], axis["direction"][1] * scale],
                z=[location[2], axis["direction"][2] * scale],
                mode="lines",
                line=dict(color=color, width=width),
                name=axis["axis"],
                hoverinfo="text",
                text=axis["axis"],
                opacity=opacity,
            )
        )
    return traces


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


def build_hover_text(name, magnitude, units=None, labels=None) -> str:
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
    if not magnitude[3:].any():
        # If all moment components are zero, only include the force components
        labels = labels[:3]
        magnitude = magnitude[:3]
    return f"{name}<br>" + "<br>".join(
        f"{label}: {value:.2f} {unit}"
        for label, value, unit in zip(labels, magnitude, units)
    )


def build_single_arrow(
    start,
    end,
    color="red",
    width=4,
    arrow_size=0.4,
    name="Arrow",
    text="",
    line_trim=0.01,
):
    """
    Creates a single straight arrow (line and cone).

    Args:
        start (np.ndarray): Start point of the arrow.
        end (np.ndarray): End point of the arrow.
        color (str): Color of the arrow.
        width (int): Width of the arrow line.
        arrow_size (float): Size of the arrowhead.
        name (str): Name of the arrow trace.
        text (str): Hover text for the arrow.
        line_trim (float): Trim factor for the arrow line. Terminate the line slightly before the tip of the arrowhead

    Returns:
        list[go.Scatter3d, go.Cone]: Plotly traces for the arrow.
    """
    vector = end - start
    return [
        go.Scatter3d(
            x=[start[0], end[0] * (1 - line_trim)],
            y=[start[1], end[1] * (1 - line_trim)],
            z=[start[2], end[2] * (1 - line_trim)],
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            hoverinfo="text",
            text=text,
        ),
        go.Cone(
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
        ),
    ]


def build_double_arrow(
    start,
    end,
    color="red",
    width=4,
    arrow_size=0.4,
    name="DoubleArrow",
    text="",
    ratio=0.9,
):
    """
    Creates a double-ended arrow by splitting the vector into two halves.

    Args:
        start (np.ndarray): Start point of the arrow.
        end (np.ndarray): End point of the arrow.
        color (str): Color of the arrow.
        width (int): Width of the arrow line.
        arrow_size (float): Size of the arrowheads.
        name (str): Name of the arrow trace.
        text (str): Hover text for the arrow.

    Returns:
        list[go.Scatter3d, go.Cone]: Plotly traces for the double arrow.
    """
    mid = start + (end - start) * ratio
    return build_single_arrow(
        start,
        mid,
        color=color,
        width=width,
        arrow_size=arrow_size,
        name=name,
        text=text,
        line_trim=0,
    ) + build_single_arrow(
        mid, end, color=color, width=width, arrow_size=arrow_size, name=name, text=text
    )


def build_curved_arrow(
    location,
    axis,
    radius,
    start_angle,
    end_angle,
    color="red",
    resolution=100,
    name="CurvedArrow",
    text="",
):
    """
    Creates a curved arrow trace to represent moments.

    Args:
        location (np.ndarray): Center point of the curved arrow.
        axis (dict): Axis dictionary with 'direction' and 'normal' keys.
        radius (float): Radius of the curved arrow.
        start_angle (float): Starting angle in radians.
        end_angle (float): Ending angle in radians.
        color (str): Color of the arrow.
        resolution (int): Number of points for the curve.
        name (str): Name of the arrow trace.
        text (str): Hover text for the arrow.

    Returns:
        list[go.Scatter3d, go.Cone]: Plotly traces for the curved arrow.
    """
    theta = np.linspace(start_angle, end_angle, resolution)
    direction = np.array(axis["direction"])
    normal = np.array(axis["normal"])

    # Validate inputs
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError("Direction vector must be non-zero.")
    if np.linalg.norm(normal) < 1e-6:
        normal = np.cross(direction, [1, 0, 0])
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(direction, [0, 1, 0])

    normal = normal / np.linalg.norm(normal)

    arc_points = (
        location
        + radius * np.outer(np.cos(theta), normal)
        + radius * np.outer(np.sin(theta), np.cross(direction, normal))
    )
    tangent_vector = arc_points[-1] - arc_points[-5]
    return [
        go.Scatter3d(
            x=arc_points[:-5, 0],
            y=arc_points[:-5, 1],
            z=arc_points[:-5, 2],
            mode="lines",
            line=dict(color=color, width=4),
            name=name,
            hoverinfo="text",
            text=text,
        ),
        go.Cone(
            x=[arc_points[-1, 0]],
            y=[arc_points[-1, 1]],
            z=[arc_points[-1, 2]],
            u=[tangent_vector[0]],
            v=[tangent_vector[1]],
            w=[tangent_vector[2]],
            sizemode="absolute",
            sizeref=radius * 0.4,
            anchor="tip",
            colorscale=[[0, color], [1, color]],
            name=f"{name} Tip",
            hoverinfo="text",
            text=text,
        ),
    ]


def build_vector_traces(
    location,
    vector,
    arrow_type: ArrowType,
    scale=1.0,
    as_components=False,
    component_colors=None,
    text=None,
    color="red",
):
    """
    Creates Plotly traces for a vector using the specified arrow type.

    Args:
        location (np.ndarray): Starting point of the vector.
        vector (np.ndarray): Vector components (x, y, z).
        arrow_type (ArrowType): Type of arrow to create.
        scale (float): Scale factor for vector length.
        as_components (bool): Whether to split the vector into x, y, z components.
        component_colors (list[str], optional): Colors for x, y, z components.
                                                Defaults to ["red", "green", "blue"].
    Returns:
        list: Plotly traces for the vector.
    """
    vector = ensure_minimum_vector(vector, scale, threshold=1.0)
    location = np.asarray(location)
    component_colors = component_colors or ["red", "green", "blue"]
    text = text or build_hover_text(
        "Vector", vector, labels=["X", "Y", "Z"], units=["", "", ""]
    )
    arrow_size = max(0.4 * scale, 0.2)
    traces = []

    if as_components:
        # Define components and colors
        components = [
            (
                vector[0] * np.array([1, 0, 0]),
                component_colors[0],
                "X-Component",
                (0, 1, 0),
            ),
            (
                vector[1] * np.array([0, 1, 0]),
                component_colors[1],
                "Y-Component",
                (0, 0, 1),
            ),
            (
                vector[2] * np.array([0, 0, 1]),
                component_colors[2],
                "Z-Component",
                (1, 0, 0),
            ),
        ]
        for comp_vector, color, name, normal in components:
            if np.linalg.norm(comp_vector) > 1e-3:  # Only add non-zero components
                if arrow_type == ArrowType.SINGLE_ARROW:
                    traces += build_single_arrow(
                        location,
                        location + comp_vector,
                        color=color,
                        name=name,
                        text=text,
                        arrow_size=arrow_size,
                    )
                elif arrow_type == ArrowType.DOUBLE_ARROW:
                    traces += build_double_arrow(
                        location,
                        location + comp_vector,
                        color=color,
                        name=name,
                        text=text,
                        arrow_size=arrow_size,
                    )
                elif arrow_type == ArrowType.CURVED_ARROW:
                    traces += build_curved_arrow(
                        location,
                        axis={
                            "direction": comp_vector / np.linalg.norm(comp_vector),
                            "normal": np.cross(comp_vector, normal),
                        },
                        radius=scale,
                        start_angle=0.2 * np.pi,
                        end_angle=2.1 * np.pi,
                        color=color,
                        name=name,
                        text=text,
                    )
    else:
        # Plot the full vector as a single arrow
        if arrow_type == ArrowType.SINGLE_ARROW:
            traces = build_single_arrow(
                location,
                location + vector,
                text=text,
                color=color,
                arrow_size=arrow_size,
            )
        elif arrow_type == ArrowType.DOUBLE_ARROW:
            traces = build_double_arrow(
                location,
                location + vector,
                text=text,
                color=color,
                arrow_size=arrow_size,
            )
        elif arrow_type == ArrowType.CURVED_ARROW:
            raise ValueError("Curved arrows are not supported for generic vectors.")
        else:
            raise ValueError(f"Unsupported ArrowType: {arrow_type}")

    return traces


def build_load_traces(
    load: Load,
    force_arrow_type: ArrowType = ArrowType.SINGLE_ARROW,
    moment_arrow_type: ArrowType = ArrowType.DOUBLE_ARROW,
    scale=1.0,
    as_components=False,
    color="red",
):
    """
    Builds Plotly traces for forces and moments with configurable arrow types.

    Args:
        location (np.ndarray): Starting point of the vectors.
        forces (np.ndarray): Force components (Fx, Fy, Fz).
        moments (np.ndarray): Moment components (Mx, My, Mz).
        force_arrow_type (ArrowType): Arrow type for forces.
        moment_arrow_type (ArrowType): Arrow type for moments.
        scale (float): Scale factor for vector lengths.
        as_components (bool): Whether to split forces and moments into components.

    Returns:
        list: A list of Plotly traces for forces and moments.
    """
    traces = []
    location = np.asarray(load.location)
    forces = np.asarray(load.magnitude[:3])
    moments = np.asarray(load.magnitude[3:])
    text = build_hover_text(load.name, load.magnitude)

    # Build force traces
    if np.linalg.norm(forces) > 1e-3:
        traces += build_vector_traces(
            location,
            forces,
            arrow_type=force_arrow_type,
            scale=scale,
            as_components=as_components,
            text=text,
            color=color,
        )

    # Build moment traces
    if np.linalg.norm(moments) > 1e-3:
        traces += build_vector_traces(
            location,
            moments,
            arrow_type=moment_arrow_type,
            scale=scale,
            as_components=as_components,
            text=text,
            color=color,
        )

    return traces


def build_mesh_traces(nodes: List[np.ndarray], faces: List[List[int]], color="blue"):
    """
    Build a Plotly Mesh3d trace for a given set of nodes and faces.

    Args:
        nodes (List[np.ndarray]): A list of node coordinates (vertices).
        faces (List[List[int]]): A list of faces, where each face is a list of three vertex indices.
        color (str): The color of the mesh.

    Returns:
        go.Mesh3d: A Plotly Mesh3d trace.
    """
    # Unpack node coordinates into x, y, z lists
    nodes = np.array(nodes)
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    # Unpack faces into i, j, k indices
    faces = np.array(faces)
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # Create a Mesh3d trace
    mesh_trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=0.5,
        name="Mesh",
    )
    return mesh_trace


def build_sphere_geometry(
    location: np.ndarray,
    radius: float,
    resolution=16,
):
    d = np.pi / resolution

    theta, phi = np.mgrid[0 : np.pi + d : d, 0 : 2 * np.pi : d]

    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.vstack([x.ravel(), y.ravel(), z.ravel()]) + location[:, None]


def build_sphere(
    location: np.ndarray,
    radius: float,
    color="blue",
    name="Sphere",
    opacity=0.5,
    alphahull=0,
    resolution=8,
    text="",
):
    x, y, z = build_sphere_geometry(location, radius, resolution=resolution)

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        color=color,
        opacity=opacity,
        alphahull=alphahull,
        name=name,
        hoverinfo="text",
        text=text,
        lighting=dict(ambient=0.5, specular=1.0),
    )


def build_mass_spheres(parts: List[Part], color="blue", scale=1.0):
    """
    Build Plotly traces for a list of Part objects with mass spheres.

    Args:
        parts (List[Part]): List of Part objects.
        color (str): Default color for the mass spheres.
        scale (float): Scale factor for the mass sphere size.

    Returns:
        List[go.Scatter3d]: A list of Plotly Scatter3d traces.
    """
    position = [part.cog for part in parts]
    size = [part.mass for part in parts]
    size = np.asarray(size) * scale / max(size)
    texts = [f"{part.id}: {part.mass:.2f} kg" for part in parts]
    traces = [
        build_sphere(center, radius, color=color, text=text)
        for center, radius, text in zip(position, size, texts)
    ]
    return traces


def build_part_traces(
    parts: List[Part], color="blue", mass_scale=1.0, load_scale=1.0, reaction_scale=1.0
):
    """
    Build Plotly traces for a list of Part objects.

    Args:
        parts (List[Part]): List of Part objects.
        color (str): Default color for the mesh.

    Returns:
        List[go.Mesh3d]: A list of Plotly Mesh3d traces.
    """
    traces = []
    # for part in parts:
    # Build mesh traces using Part nodes and faces
    # traces.append(build_mesh_traces(part.nodes, part.faces, color=color))
    traces.extend(build_mass_spheres(parts, color=color, scale=mass_scale))

    # Build load and reaction traces
    max_load = max(
        max(np.abs(load.magnitude).max(), np.abs(reaction.magnitude).max())
        for part in parts
        for load in part.loads
        if isinstance(load, Load)
        for reaction in part.reactions
        if isinstance(reaction, Reaction)
    )

    traces.extend(
        build_load_traces(load, scale=load_scale / max_load, color="red")
        for part in parts
        for load in part.loads
        if isinstance(load, Load)
    )

    traces.extend(
        build_load_traces(reaction, scale=reaction_scale / max_load, color="green")
        for part in parts
        for reaction in part.reactions
        if isinstance(reaction, Reaction)
    )
    # flatten the list of lists
    traces = [
        trace
        for sublist in traces
        for trace in (sublist if isinstance(sublist, list) else [sublist])
    ]
    return traces


import numpy as np


def compute_bounds_from_traces(traces, padding=0):
    """
    Computes the bounds of a set of Plotly 3D traces.

    Args:
        traces (list): List of Plotly traces (e.g., Scatter3d, Mesh3d, etc.).

    Returns:
        tuple: ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    """
    all_points = []

    for trace in traces:
        if isinstance(trace, go.Scatter3d):
            # Extract x, y, z from Scatter3d
            x, y, z = trace.x, trace.y, trace.z
            if x and y and z:  # Ensure valid data
                all_points.append(np.vstack([x, y, z]).T)

        elif isinstance(trace, go.Mesh3d):
            # Extract x, y, z from Mesh3d
            x, y, z = trace.x, trace.y, trace.z
            if x.any() and y.any() and z.any():
                all_points.append(np.vstack([x, y, z]).T)

        elif isinstance(trace, go.Surface):
            # Extract x, y, z from Surface
            x, y, z = trace.x, trace.y, trace.z
            if isinstance(z, np.ndarray):
                # If Surface, x and y might be grid-based
                x, y = np.meshgrid(x, y)
                all_points.append(np.vstack([x.ravel(), y.ravel(), z.ravel()]).T)

        # Add other trace types as needed

    if not all_points:
        raise ValueError("No valid data points found in traces.")

    # Combine all points and compute bounds
    all_points = np.vstack(all_points)
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    min_z, max_z = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    # Apply padding
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    min_z -= padding
    max_z += padding

    return ((min_x, max_x), (min_y, max_y), (min_z, max_z))


# def build_arrow(
#     start, end, name="Arrow", color="blue", width=4, text="", arrow_size=0.4
# ):
#     """
#     Creates a 3D arrow with a line and a cone.

#     Args:
#         start (np.ndarray): Start point of the arrow.
#         end (np.ndarray): End point of the arrow.
#         name (str): Name of the arrow group.
#         color (str): Color of the arrow.
#         width (int): Width of the arrow line.
#         text (str): Hover text for the arrow.
#         arrow_size (float): Size reference for the arrowhead.

#     Returns:
#         list: A list of Plotly traces (line and cone).
#     """
#     start = np.asarray(start)
#     end = np.asarray(end)
#     vector = end - start

#     # Line (vector body)
#     body = go.Scatter3d(
#         x=[start[0], end[0] * 0.99],
#         y=[start[1], end[1] * 0.99],
#         z=[start[2], end[2] * 0.99],
#         mode="lines",
#         line=dict(color=color, width=width),
#         name=name,
#         hoverinfo="text",
#         text=text,
#     )

#     # Cone (vector tip)
#     tip = go.Cone(
#         x=[end[0]],
#         y=[end[1]],
#         z=[end[2]],
#         u=[vector[0]],
#         v=[vector[1]],
#         w=[vector[2]],
#         sizemode="absolute",
#         sizeref=arrow_size,
#         anchor="tip",
#         colorscale=[[0, color], [1, color]],
#         name=name,
#         hoverinfo="text",
#         text=text,
#     )

#     return [body, tip]


# def build_load_traces(
#     item, name=None, scale=1.0, as_components=False, labels=None, color="blue"
# ) -> list:
#     """
#     Creates Plotly traces for both force and moment vectors from a BoundVector object.

#     Args:
#         item (BoundVector): A vector object containing location and magnitude.
#         name (str, optional): Name of the vector for display. Defaults to None.
#         scale (float, optional): Scale factor for vector length. Defaults to 1.0.
#         as_components (bool, optional): Whether to split vectors into components. Defaults to False.
#         labels (list, optional): Custom labels for vector components. Defaults to None.
#         color (str, optional): Color of the vector traces. Defaults to "blue".

#     Returns:
#         list: A list of Plotly traces.
#     """
#     if not isinstance(item, BoundVector):
#         raise TypeError(f"Expected a BoundVector, got {type(item).__name__}")

#     name = name or item.name or "Unnamed Vector"
#     text = generate_hover_text(name, item.magnitude, labels)

#     # Build force traces
#     forces = build_force_traces(
#         location=item.location,
#         magnitude=item.magnitude[:3],
#         name=name,
#         scale=scale,
#         as_components=as_components,
#         text=text,
#         color=color,
#     )

#     # Build moment traces
#     moments = build_moment_traces(
#         location=item.location,
#         moments=item.magnitude[3:],
#         name=name,
#         scale=scale,
#         text=text,
#         color=color,
#         as_components=as_components,
#     )

#     return forces + moments


# def build_force_traces(
#     location: np.ndarray,
#     magnitude: np.ndarray,
#     name: str = "Vector",
#     scale: float = 1.0,
#     color: str = "blue",
#     width: int = 4,
#     as_components: bool = False,
#     component_colors=["red", "green", "blue"],
#     text="",
#     arrow_size=0.4,
# ):
#     """
#     Builds Plotly traces for 3D force vectors, optionally split into components.

#     Args:
#         location (np.ndarray): Starting point of the vector.
#         magnitude (np.ndarray): Magnitude of the vector components.
#         name (str): Name of the vector trace.
#         scale (float): Scaling factor for vector length.
#         color (str): Color for the vector body.
#         width (int): Width of the vector line.
#         as_components (bool): Whether to split the vector into XYZ components.
#         component_colors (list): Colors for X, Y, Z components.
#         text (str): Hover text for the vector.
#         arrow_size (float): Size reference for arrowheads.

#     Returns:
#         list: Plotly traces for the vector.
#     """
#     location = np.asarray(location)
#     magnitude = np.asarray(magnitude)

#     if as_components:
#         components = [
#             (location, location + np.array([magnitude[0], 0, 0]) * scale),
#             (location, location + np.array([0, magnitude[1], 0]) * scale),
#             (location, location + np.array([0, 0, magnitude[2]]) * scale),
#         ]
#         traces = []
#         for color, (start, end) in zip(component_colors, components):
#             traces.extend(
#                 build_arrow(
#                     start,
#                     end,
#                     color=color,
#                     text=text,
#                     name=name,
#                     width=width,
#                     arrow_size=arrow_size,
#                 )
#             )
#         return traces

#     return build_arrow(
#         location,
#         location + magnitude * scale,
#         color=color,
#         text=text,
#         name=name,
#         arrow_size=arrow_size,
#     )


# def build_moment_traces(
#     location: np.ndarray,
#     moments: np.ndarray,
#     scale: float = 5,
#     name: str = "Moments",
#     color: str = "red",
#     width: int = 4,
#     as_components: bool = False,
#     component_colors=["red", "green", "blue"],
#     text="",
#     base_radius: float = 2,
#     start_offset: float = 0.3,
#     end_offset: float = 1.1,
#     resolution: int = 100,
#     arrow_base_ref: float = 0.15,
#     tolerance: float = 1e-3,
# ):
#     """
#     Builds Plotly traces for 3D rotational moments, optionally split into components.

#     Args:
#         location (np.ndarray): Center point of rotation.
#         moments (np.ndarray): Moment magnitudes about X, Y, Z axes.
#         scale (float): Scaling factor for arc size.
#         name (str): Name of the trace group.
#         color (str): Color of the arcs and cones.
#         width (int): Line width for arcs.
#         as_components (bool): Whether to split moments into X, Y, Z components.

#     Returns:
#         list: Plotly traces for rotational moments.
#     """

#     def build_curved_arrow(location, axis, text, color):
#         theta = np.linspace(
#             start_offset * 2 * np.pi, end_offset * 2 * np.pi, resolution
#         )
#         direction = np.array(axis["direction"])
#         normal = np.array(axis["normal"])
#         radius = base_radius * scale

#         arc_points = (
#             location
#             + radius * np.outer(np.cos(theta), normal)
#             + radius * np.outer(np.sin(theta), np.cross(direction, normal))
#         )

#         traces = [
#             go.Scatter3d(
#                 x=arc_points[:-5, 0],
#                 y=arc_points[:-5, 1],
#                 z=arc_points[:-5, 2],
#                 mode="lines",
#                 line=dict(color=color, width=width),
#                 name=name,
#                 hoverinfo="text",
#                 text=text,
#             )
#         ]
#         arrow_tip = arc_points[-1]
#         tangent_start_idx = int(len(arc_points) * arrow_base_ref)
#         tangent_vector = arc_points[-1] - arc_points[-tangent_start_idx]
#         traces.append(
#             go.Cone(
#                 x=[arrow_tip[0]],
#                 y=[arrow_tip[1]],
#                 z=[arrow_tip[2]],
#                 u=[tangent_vector[0]],
#                 v=[tangent_vector[1]],
#                 w=[tangent_vector[2]],
#                 sizemode="absolute",
#                 sizeref=radius * 0.8,
#                 colorscale=[[0, color], [1, color]],
#                 anchor="tip",
#                 name=f"{name} Tip",
#                 hoverinfo="text",
#                 text=text,
#             )
#         )
#         return traces

#     traces = []
#     location = np.asarray(location)
#     moments = np.asarray(moments)

#     if as_components:
#         for (i, axis), color in zip(enumerate(PLOT_AXES), component_colors):
#             if moments[i] >= tolerance:
#                 traces += build_curved_arrow(location, axis, text, color)
#     else:
#         if np.linalg.norm(moments) >= tolerance:
#             axis = {
#                 "direction": _convert_unit_vector(moments),
#                 "normal": _convert_unit_vector(np.cross(moments, [0, 0, 1])),
#             }
#             traces += build_curved_arrow(location, axis, text, color)

#     return traces
