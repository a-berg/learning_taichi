import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

# resolution of the problem
W, H = 256, 256
# for initialization purposes
np_grid = np.zeros((W, H, 2), dtype=np.float32)
np_grid[:, :, 0] = 1.0  # Reactant H = 1.0 in all domain initially.
# square with reactant V = 1.0 in the middle of the domain
np_grid[(W // 2 - 10) : (W // 2 + 10), (H // 2 - 10) : (H // 2 + 10), 1] = 1.0

domain = ti.Vector.field(n=2, dtype=ti.f32, shape=(W, H))

# auxiliary field for PDE solving
future = ti.Vector.field(n=2, dtype=ti.f32, shape=(W, H))

# used for rendering
pixels = ti.field(dtype=ti.f32, shape=(W, H))

# Define constants
r_u: float = 0.250
r_v: float = r_u / 2  # 0.080
feed: float = 0.040
kill: float = 0.062

@ti.func
def laplacian(i: int, j: int):
    """Compute the laplacian of a point identified by i and j.

    This Taichi function simply computes the discrete laplacian over a regular grid by
    using finite differences.

    Parameters
    ----------
    i : int
        reference to the first index of the point in the grid.
    j : int
        reference to the second intex of the point in the grid
    """
    return (
        domain[i + 1, j]
        + domain[i, j + 1]
        + domain[i - 1, j]
        + domain[i, j - 1]
        - domain[i, j] * 4.0
    )

@ti.kernel
def render():
    """Differently to a scalar field, vector fields need to be processed a bit for them
    to be paintable."""
    for i, j in domain:
        # paint just the V concentration.
        pixels[i, j] = domain[i, j][1]

@ti.kernel
def evolve():
    """Integrate one timestep of the discretized Gray-Scott equation.

    Define a Taichi kernel to compute the next state of the system. Uses Explicit Euler
    to integrate.
    """
    for i, j in domain:
        uv = domain[i, j]
        reaction = uv[0] * uv[1] * uv[1]
        Δ = laplacian(i, j)
        du = r_u * Δ[0] - reaction + feed * (1 - uv[0])
        dv = r_v * Δ[1] + reaction - (feed + kill) * uv[1]
        # instead of returning, update in place (returning would get us outside
        # of the GPU)
        uv_1 = uv + 0.5 * tm.vec2(du, dv)
        future[i, j] = uv_1

    for I in ti.grouped(domain):
        domain[I] = future[I]

def main():
    gui = ti.GUI("Gray Scott", res=(W, H))
    substeps: int = 60  # 1
    domain.from_numpy(np_grid)
    result_dir = "./results"
    # VideoManager let's me create gifs easily.
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        # If we compute each time we render, the system evolves very slowly.
        # By evolving the equation 60 times before rendering, we accelerate the real
        # time evolution.
        for _ in range(substeps):
            evolve()
        # # canvas.set_image(domain)
        render()
        gui.set_image(pixels)
        gui.show()
        video_manager.write_frame(pixels.to_numpy())

    video_manager.make_video(gif=False)  # output as mp4 that will be converted to gif later


# wrapping thing into `if __name__=="__main__"` prevents the function from being
# executed if we call `ti run` (my preferred method to run taichi code) in the command
# line, because that way this file is no longer "__main__".
main()
