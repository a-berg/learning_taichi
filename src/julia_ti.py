import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))


@ti.func
def complex_square(z):  # complex square of a 2D vector
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])


@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = tm.vec2(0.7885 * tm.cos(t), 0.7885 * tm.sin(t))
        z = tm.vec2(i / n - 1, j / n - 0.5) * 3
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_square(z) + c
            iterations += 1
            pixels[i, j] = 1 - iterations * 0.02


gui = ti.GUI("Julia Set", res=(n * 2, n))
video_manager = ti.tools.VideoManager(
    output_dir="./results/julia/", framerate=24, automatic_build=False
)

τ = 6.28318530718
max_iter = 1_000
for i in range(max_iter):
    paint(i * τ / 2 / (n - 1))
    gui.set_image(pixels)
    video_manager.write_frame(pixels.to_numpy())
    gui.show()

video_manager.make_video(gif=True, mp4=True)
