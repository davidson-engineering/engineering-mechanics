import math
import os
import sys

import glm
import moderngl
import pygame
from objloader import Obj
from PIL import Image

os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"

pygame.init()
pygame.display.set_mode((800, 800), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)


class ImageTexture:
    def __init__(self, path):
        self.ctx = moderngl.get_context()

        img = Image.open(path).convert("RGBA")
        self.texture = self.ctx.texture(img.size, 4, img.tobytes())
        self.sampler = self.ctx.sampler(texture=self.texture)

    def use(self):
        self.sampler.use()


class ModelGeometry:
    def __init__(self, path):
        self.ctx = moderngl.get_context()

        obj = Obj.open(path)
        self.vbo = self.ctx.buffer(obj.pack("vx vy vz nx ny nz tx ty"))

    def vertex_array(self, program):
        return self.ctx.vertex_array(
            program, [(self.vbo, "3f 3f 2f", "in_vertex", "in_normal", "in_uv")]
        )


class Mesh:
    def __init__(self, program, geometry, texture=None):
        self.ctx = moderngl.get_context()
        self.vao = geometry.vertex_array(program)
        self.texture = texture

    def render(self, position, color, scale):
        self.vao.program["use_texture"] = False

        if self.texture:
            self.vao.program["use_texture"] = True
            self.texture.use()

        self.vao.program["position"] = position
        self.vao.program["color"] = color
        self.vao.program["scale"] = scale
        self.vao.render()


import moderngl
import moderngl_window as mglw


class TestRenderer(mglw.WindowConfig):
    gl_version = (4, 1)
    title = "Minimal Test"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.program = self.ctx.program(
            vertex_shader="""
            #version 330 core
            layout(location = 0) in vec3 in_position;
            void main() {
                gl_Position = vec4(in_position, 1.0);
            }
            """,
            fragment_shader="""
            #version 330 core
            out vec4 fragColor;
            void main() {
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red color
            }
            """,
        )

        self.texture = ImageTexture("examples/data/textures/crate.png")

        self.car_geometry = ModelGeometry("examples/data/models/lowpoly_toy_car.obj")
        self.car = Mesh(self.program, self.car_geometry)

        self.crate_geometry = ModelGeometry("examples/data/models/crate.obj")
        self.crate = Mesh(self.program, self.crate_geometry, self.texture)

    def camera_matrix(self):
        now = pygame.time.get_ticks() / 1000.0
        eye = (math.cos(now), math.sin(now), 0.5)
        proj = glm.perspective(45.0, 1.0, 0.1, 1000.0)
        look = glm.lookAt(eye, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        return proj * look

    def render(self):
        camera = self.camera_matrix()

        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        self.program["camera"].write(camera)
        self.program["light_direction"] = (1.0, 2.0, 3.0)

        self.car.render((-0.4, 0.0, 0.0), (1.0, 0.0, 0.0), 0.2)
        self.crate.render((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.2)
        self.car.render((0.4, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2)


mglw.run_window_config(TestRenderer)
