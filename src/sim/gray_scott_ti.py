from itertools import cycle
import taichi as ti
import taichi.math as tm
import numpy as np
from dataclasses import dataclass, astuple
from typing import Optional

ti.init(arch=ti.gpu)

VIDEO_OUT = False


@dataclass
class DomainShape:
    width: int
    height: int


@dataclass
class DiffusionCoefficients:
    """Diffusion coefficients.

    Ideally it should be a vector."""

    u: float
    v: float


@dataclass
class ReactionCoefficients:
    """Reaction coefficients."""

    feed: float
    kill: float


@ti.data_oriented
class GrayScottProblem:
    """Class representing the G-S problem to be solved.

    For now it's not generic at all. Should be in its own file tbh.
    """

    def __init__(
        self,
        domain_shape: DomainShape,
        diffusion_coefs: DiffusionCoefficients,
        reaction_coefs: ReactionCoefficients,
        f_arr: Optional[np.array] = None,
        k_arr: Optional[np.array] = None,
    ):
        """Initialize parameters."""
        self.D = tm.vec2(astuple(diffusion_coefs))
        self.R = tm.vec2(astuple(reaction_coefs))
        self.domain_shape = domain_shape
        # these arrays will allow for space-varying coefficients.
        self.f_arr = f_arr if f_arr is not None else np.ones(astuple(domain_shape))
        self.k_arr = k_arr if k_arr is not None else np.ones(astuple(domain_shape))
        self.allocate_domain()

    def allocate_domain(self):
        W, H = astuple(self.domain_shape)
        np_grid = np.zeros((2, W, H, 2), dtype=np.float32)
        np_grid[0, :, :, 0] = 1.0  # Reactant H = 1.0 in all domain initially.
        # square with reactant V = 1.0 in the middle of the domain
        np_grid[
            0, (W // 2 - 10) : (W // 2 + 10), (H // 2 - 10) : (H // 2 + 10), 1
        ] = 1.0

        self.domain = ti.Vector.field(n=2, dtype=ti.f32, shape=(2, W, H))
        self.domain.from_numpy(np_grid)

    @ti.kernel
    def evolve(self, t: int):
        """Integrate one timestep of the discretized Gray-Scott equation.

        Define a Taichi kernel to compute the next state of the system. Uses Explicit Euler
        to integrate.
        """
        A = ti.Matrix([[1, 0], [-1, -1]])
        for i, j in ti.ndrange(*astuple(self.domain_shape)):
            uv = self.domain[t, i, j]
            reaction = self._reaction(uv)
            reaction *= tm.vec2([-1, 1])
            Δ = self.laplacian(t, i, j)
            du = self.D.u * Δ[0] - reaction + self.R.feed * (1 - uv[0])
            dv = self.D.v * Δ[1] + reaction - (self.R.feed + self.R.kill) * uv[1]
            # attempting to vectorize this
            duv = self.D * Δ + reaction + (A * self.R) @ (tm.vec2([1, 0]) - uv)
            uv_1 = uv + 0.5 * tm.vec2(du, dv)  # is h=0.5 for some reason?
            self.domain[1 - t, i, j] = uv_1

    @ti.func
    def _reaction(self, uv):
        return uv[0] * uv[1] * uv[1]

    @ti.func
    def laplacian(self, t: int, i: int, j: int):
        """Compute the laplacian of a point identified by i and j.

        This Taichi function simply computes the discrete laplacian over a regular grid
        by using finite differences.

        Parameters
        ----------
        t : int
            reference to alternating buffer.
        i : int
            reference to the first index of the point in the grid.
        j : int
            reference to the second intex of the point in the grid

        """
        return (
            self.domain[t, i + 1, j]
            + self.domain[t, i, j + 1]
            + self.domain[t, i - 1, j]
            + self.domain[t, i, j - 1]
            - self.domain[t, i, j] * 4.0
        )


@ti.data_oriented
class ColorRenderer:
    def __init__(self, domain_shape: DomainShape):
        self.pixels = ti.Vector.field(
            3, ti.f32, shape=(domain_shape.width, domain_shape.height)
        )
        self.palette = ti.Vector.field(4, ti.f32, shape=(5,))
        self.palette[0] = [0.0, 0.0, 0.0, 0.3137]
        self.palette[1] = [1.0, 0.1843, 0.53333, 0.37647]
        self.palette[2] = [0.8549, 1.0, 0.53333, 0.388]
        self.palette[3] = [0.376, 1.0, 0.478, 0.392]
        self.palette[4] = [1.0, 1.0, 1.0, 1]

    @ti.kernel
    def render(self, domain: ti.template()):
        for i, j in self.pixels:
            value = domain[0, i, j].y
            c = tm.vec3(value)
            # clamp value
            if value <= self.palette[0].w:
                c = self.palette[0].xyz

            for k in range(4):
                c0 = self.palette[k]
                c1 = self.palette[k + 1]
                if c0.w < value < c1.w:
                    a = (value - c0.w) / (c1.w - c0.w)
                    c = tm.mix(c0.xyz, c1.xyz, a)

            self.pixels[i, j] = c


def main():
    domain_shape = DomainShape(600, 400)
    d = DiffusionCoefficients(u=0.250, v=0.125)
    r = ReactionCoefficients(feed=0.052, kill=0.062)
    problem = GrayScottProblem(domain_shape, d, r)
    renderer = ColorRenderer(domain_shape)
    # Use Window instead of GUI
    gui = ti.ui.Window("Gray Scott", res=astuple(domain_shape))
    # get a canvas to paint on
    canvas = gui.get_canvas()
    substeps: int = 30  # 1
    result_dir = "./results/reaction_diffusion/"
    # VideoManager lets me create gifs easily.
    if VIDEO_OUT:
        video_manager = ti.tools.VideoManager(
            output_dir=result_dir, framerate=24, automatic_build=False
        )
    while gui.running:
        for _, t in zip(range(2 * substeps), cycle([0, 1])):
            problem.evolve(t)
        renderer.render(problem.domain)
        canvas.set_image(renderer.pixels)
        gui.show()
        if VIDEO_OUT:
            video_manager.write_frame(renderer.pixels.to_numpy())

    if VIDEO_OUT:
        video_manager.make_video(gif=True)


main()
