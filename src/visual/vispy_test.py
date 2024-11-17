from vispy import app, gloo
from vispy import app

app.use_app("glfw")

from vispy.gloo import Program
import numpy as np

import logging

logging.basicConfig(level=logging.DEBUG)


# Vertex shader
VERTEX_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
attribute vec3 a_color;
varying vec3 v_color;
void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

# Fragment shader
FRAGMENT_SHADER = """
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
"""

# Define tetrahedron geometry
VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.5, 1.0, 0.0],  # Vertex 2
        [0.5, 0.5, 1.0],  # Vertex 3
    ],
    dtype=np.float32,
)

COLORS = np.array(
    [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
    ],
    dtype=np.float32,
)

FACETS = np.array(
    [
        [0, 1, 2],  # Triangle 1
        [0, 1, 3],  # Triangle 2
        [1, 2, 3],  # Triangle 3
        [0, 2, 3],  # Triangle 4
    ],
    dtype=np.uint32,
)


class TetrahedronCanvas(app.Canvas):
    def __init__(self):
        super().__init__(
            size=(800, 600), title="Tetrahedron Example", keys="interactive"
        )
        self.program = Program(VERTEX_SHADER, FRAGMENT_SHADER)

        # Flatten vertex and color data for indexing
        vertex_data = VERTICES[FACETS.flatten()]
        color_data = COLORS[FACETS.flatten()]

        # Debugging: Print vertex and color data
        print("Vertex Data:\n", vertex_data)
        print("Color Data:\n", color_data)

        # Pass data to GPU
        self.program["a_position"] = gloo.VertexBuffer(vertex_data)
        self.program["a_color"] = gloo.VertexBuffer(color_data)

        # Set up transformation matrices
        self.model = np.eye(4, dtype=np.float32)
        self.view = self._create_view_matrix(
            np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])
        )
        self.projection = self._create_perspective_matrix(45, 800 / 600, 0.1, 10.0)

        # Debugging: Print matrices
        print("Model Matrix:\n", self.model)
        print("View Matrix:\n", self.view)
        print("Projection Matrix:\n", self.projection)

        self.program["u_model"] = self.model
        self.program["u_view"] = self.view
        self.program["u_projection"] = self.projection

        gloo.set_clear_color((0.1, 0.1, 0.1, 1.0))
        # gloo.set_state(depth_test=True)
        gloo.set_state(depth_test=False, cull_face=False)

    # def on_draw(self, event):
    #     gloo.clear(color="blue")
    #     print("Basic frame drawn.")

    def on_draw(self, event):
        gloo.clear(color="w", depth=True)
        # Debugging: Print draw call information
        print("Drawing frame...")
        self.program.draw("triangles")

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)
        aspect_ratio = event.size[0] / event.size[1]
        self.projection = self._create_perspective_matrix(45, aspect_ratio, 0.1, 10.0)
        self.program["u_projection"] = self.projection

    @classmethod
    def _create_view_matrix(cls, eye, target, up):
        f = (target - eye) / np.linalg.norm(target - eye)
        r = np.cross(up, f) / np.linalg.norm(np.cross(up, f))
        u = np.cross(f, r)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.stack([r, u, -f], axis=-1)
        mat[:3, 3] = -mat[:3, :3] @ eye
        return mat

    @classmethod
    def _create_perspective_matrix(cls, fov, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        nf = 1.0 / (near - far)
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = (far + near) * nf
        mat[2, 3] = 2 * far * near * nf
        mat[3, 2] = -1.0
        return mat


if __name__ == "__main__":
    canvas = TetrahedronCanvas()
    canvas.show()
    app.run()
