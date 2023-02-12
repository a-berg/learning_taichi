import random
from math import pi

n = 100_000_000


def estimate_pi(n):
    m = 0
    for i in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x * x + y * y <= 1:
            m += 1
    return 4 * m / n


pi_est = estimate_pi(n)
print(f"Estimate of π after {n} iterations:  {pi_est:.6f}")
print(f"Error: {abs(pi-pi_est)/pi:.7f}%")
