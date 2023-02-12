import taichi as ti
import taichi.math as tm
from math import pi

ti.init(arch=ti.cpu)

n = 10_000
pi_est = ti.field(dtype=ti.f64, shape=())


@ti.kernel
def estimate_pi(n: int):
    for i, j in ti.ndrange(n, n):
        if tm.length(tm.vec2(i, j)) <= n:
            pi_est[None] += 1.0
    return


estimate_pi(n)

pi_est = 4 / n / n * pi_est.to_numpy()
print(f"Estimate of π after {n**2} iterations:  {pi_est:.6f}")
print(f"Error: {abs(pi-pi_est)/pi:.7f}%")
