import plotly.graph_objects as go
import numpy as np

# Sample data for loads and reactions (in 3D space)
loads = [
    {
        "location": np.array([1, 1, 1]),
        "vector": np.array([100, 200, -50]),
        "name": "Load 1",
    },
    {
        "location": np.array([2, 0, 2]),
        "vector": np.array([-50, 100, 50]),
        "name": "Load 2",
    },
]

reactions = [
    {
        "location": np.array([0, 0, 0]),
        "vector": np.array([-100, -200, 50]),
        "name": "Reaction 1",
    },
    {
        "location": np.array([1, -1, 0]),
        "vector": np.array([50, -100, -50]),
        "name": "Reaction 2",
    },
]


# Function to plot vectors
def plot_vectors(vectors, color, name_prefix):
    data = []
    for vec in vectors:
        start = vec["location"]
        vector = vec["vector"]

        # Calculate end point for the vector
        end = start + vector * 0.01  # Scale down for visibility

        # Create the vector line
        data.append(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines+markers+text",
                line=dict(color=color, width=5),
                marker=dict(size=4),
                name=f"{name_prefix} - {vec['name']}",
                text=[f"{name_prefix} Start", f"{name_prefix} End"],
                textposition="top center",
            )
        )
    return data


# Create Plotly figure
fig = go.Figure()

# Plot loads as blue vectors
fig.add_traces(plot_vectors(loads, "blue", "Load"))

# Plot reactions as red vectors
fig.add_traces(plot_vectors(reactions, "red", "Reaction"))

# Customize layout
fig.update_layout(
    title="3D Load and Reaction Vectors",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
    legend=dict(title="Legend", itemsizing="constant"),
)

# Show the plot
fig.show()
