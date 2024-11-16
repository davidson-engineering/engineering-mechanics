import moderngl
import moderngl_window as mglw
import numpy as np


class VectorRenderer(mglw.WindowConfig):
    gl_version = (4, 1)  # OpenGL 3.3 or higher
    title = "3D Vector Visualization"
    window_size = (800, 600)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Shader program for vectors
        self.program = self.ctx.program(
            vertex_shader="""
            #version 410 core
            in vec3 in_position;
            in vec3 in_color;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            out vec3 color;
            void main() {
                gl_Position = projection * view * model * vec4(in_position, 1.0);
                color = in_color;
            }
            """,
            fragment_shader="""
            #version 410 core
            in vec3 color;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(color, 1.0);
            }
            """,
        )

        # Define arrow vertices and colors
        self.vertices = np.array(
            [
                [0.0, 0.0, 0.0],  # Base of the vector
                [0.0, 1.0, 0.0],  # Tip of the vector
            ],
            dtype="f4",
        )

        self.colors = np.array(
            [
                [1.0, 1.0, 0.0],  # Red base
                [1.0, 1.0, 0.0],  # Red tip
            ],
            dtype="f4",
        )

        # Create buffer objects
        self.vbo = self.ctx.buffer(self.vertices.tobytes())
        self.cbo = self.ctx.buffer(self.colors.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, "3f", "in_position"),
                (self.cbo, "3f", "in_color"),
            ],
        )

        # Matrices
        self.model = np.eye(4, dtype="f4")
        self.view = self.create_look_at(
            np.array([3, 3, 3]), np.array([0, 0, 0]), np.array([0, 1, 0])
        )
        self.projection = self.create_perspective_projection(
            45, self.aspect_ratio, 0.1, 100
        )

    def render(self, time, frametime):
        self.ctx.clear(0.1, 0.1, 0.1)  # Clear the screen with a gray color
        self.ctx.enable_only(moderngl.DEPTH_TEST)  # Enable depth testing

        # Set uniforms
        self.program["model"].write(self.model.tobytes())
        self.program["view"].write(self.view.tobytes())
        self.program["projection"].write(self.projection.tobytes())

        # Render the vector
        self.vao.render(moderngl.LINES)

    def create_look_at(self, eye, target, up):
        f = (target - eye) / np.linalg.norm(target - eye)
        r = np.cross(up, f) / np.linalg.norm(np.cross(up, f))
        u = np.cross(f, r)
        return np.array(
            [
                [r[0], u[0], f[0], 0.0],
                [r[1], u[1], f[1], 0.0],
                [r[2], u[2], f[2], 0.0],
                [-np.dot(r, eye), -np.dot(u, eye), -np.dot(f, eye), 1.0],
            ],
            dtype="f4",
        )

    def create_perspective_projection(self, fov, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        nf = 1.0 / (near - far)
        return np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) * nf, -1],
                [0, 0, (2 * far * near) * nf, 0],
            ],
            dtype="f4",
        )


if __name__ == "__main__":
    mglw.run_window_config(VectorRenderer)
