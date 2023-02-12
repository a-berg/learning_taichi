import numpy as np
import matplotlib.pyplot as plt


def evolve(domain):
    domain_ = np.pad(domain, ((0, 0), (1, 1), (1, 1)), "wrap")
    u = domain[0, :, :]
    v = domain[1, :, :]
    r = u * v * v

    #     0.05*domain_[:, :-2, :-2]\
    #   + 0.05*domain_[:, :-2, 2:]\
    #   + 0.05*domain_[:, 2:, :-2]\
    #   + 0.05*domain_[:, 2:,  2:]\

    laplacian = (
        0.25 * domain_[:, 1:-1, :-2]
        + 0.25 * domain_[:, 1:-1, 2:]
        + 0.25 * domain_[:, :-2, 1:-1]
        + 0.25 * domain_[:, 2:, 1:-1]
        - domain
    )
    du = τ * 1.0 * laplacian[0, :, :] - r + γ * (1 - u)
    dv = τ * 0.5 * laplacian[1, :, :] + r - (γ + k) * v
    return domain + np.stack((du, dv), 0)


def center_square(M, l=10):
    _, n, m = M.shape
    n //= 2
    m //= 2
    return slice(n - l, n + l), slice(m - l, m + l)


τ = 0.8388  # magic number wtf -- without this it doesn't work!
γ, k = 0.024, 0.055
# γ, k = .055, .062
resol = 256

domain = np.zeros((2, resol, resol))
# domain = np.zeros((2, resol, int(16/9*resol)))
domain[0, :, :] = 1
idx1, idx2 = center_square(domain, 10)
domain[1, idx1, idx2] = 1

for i in range(10000):
    domain = evolve(domain)

plt.figure(figsize=(16, 9))
plt.imshow(domain[1])
plt.colorbar()
