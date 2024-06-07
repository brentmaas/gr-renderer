import numpy as np
import matplotlib.pyplot as plt

def to_isotropic(x, y, z, dx, dy, dz, mass):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r_iso = (np.sqrt(r) + np.sqrt(r - 2 * mass)) ** 2 / 4
    x_iso = x / (1 + mass / 2 / r_iso) ** 2
    y_iso = y / (1 + mass / 2 / r_iso) ** 2
    z_iso = z / (1 + mass / 2 / r_iso) ** 2
    dx_iso = dx / (1 + mass / 2 / r_iso) ** 2
    dy_iso = dy / (1 + mass / 2 / r_iso) ** 2
    dz_iso = dz / (1 + mass / 2 / r_iso) ** 2
    f = mass / 2 / r_iso
    dt = np.sqrt((1 + f) ** 6 / (1 - f) ** 2 * (dx_iso ** 2 + dy_iso ** 2 + dz_iso ** 2))
    return x_iso, y_iso, z_iso, dx_iso, dy_iso, dz_iso, dt

def christoffel_symbols(x, y, z, mass):
    """
    Gamma^mu_rho_sigma -> array[mu][rho][sigma]
    mu, rho, sigma -> t, r, theta, phi
    
    From https://arxiv.org/pdf/0904.4184
    """
    
    zero = np.zeros(len(x)) if type(x) == np.ndarray else 0
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    factor = mass / r ** 3
    factor_plus = 1 + mass / r / 2
    factor_minus = 1 - mass / r / 2
    return np.nan_to_num(np.array(
    [
        [ # t
            [zero, x * factor / factor_minus, y * factor / factor_minus, z * factor / factor_minus], # t, t
            [x * factor / factor_minus, zero, zero, zero], # t, x
            [y * factor / factor_minus, zero, zero, zero], # t, y
            [z * factor / factor_minus, zero, zero, zero] # t, z
        ],
        [ # x
            [x * factor * factor_minus / factor_plus ** 7, zero, zero, zero], # x, t
            [zero, -x * factor / factor_plus, -y * factor / factor_plus, -z * factor / factor_plus], # x, x
            [zero, -y * factor / factor_plus, x * factor / factor_plus, zero], # x, y
            [zero, -z * factor / factor_plus, zero, x * factor / factor_plus] # x, z
        ],
        [ # y
            [y * factor * factor_minus / factor_plus ** 7, zero, zero, zero], # y, t
            [zero, y * factor / factor_plus, -x * factor / factor_plus, zero], # y, x
            [zero, -x * factor / factor_plus, -y * factor / factor_plus, -z * factor / factor_plus], # y, y
            [zero, zero, -z * factor / factor_plus, y * factor / factor_plus] # y, z
        ], 
        [ # z
            [z * factor * factor_minus / factor_plus ** 7, zero, zero, zero], # z, t
            [zero, z * factor / factor_plus, zero, -x * factor / factor_plus], # z, x
            [zero, zero, z * factor / factor_plus, -y * factor / factor_plus], # z, y
            [zero, -x * factor / factor_plus, -y * factor / factor_plus, -z * factor / factor_plus] # z, z
        ]
    ]))

def geodesic_leapfrog_step(pos, vel, daffine, mass):
    npos = pos + vel * daffine
    
    symbols = christoffel_symbols(pos[1], pos[2], pos[3], mass)
    dvel = np.zeros_like(vel)
    for mu in range(4):
        for rho in range(4):
            for sigma in range(4):
                dvel[mu] -= symbols[mu][rho][sigma] * vel[rho] * vel[sigma]
    nvel = vel + dvel * daffine
    
    inside_horizon = npos[1] * npos[1] + npos[2] * npos[2] + npos[3] * npos[3] <= mass ** 2 / 4
    nvel[:,inside_horizon] = vel[:,inside_horizon]
    npos[:,inside_horizon] = pos[:,inside_horizon]
    
    return npos, nvel

mass = 1
initial_pos = []
vel = []
rays = 12
# Uniformly distributed rays from (3, 0)
for i in range(rays):
    x, y, z, dx, dy, dz = 3.0, 0, 0, np.cos(2 * np.pi * i / rays), 0, np.sin(2 * np.pi * i / rays)
    x_iso, y_iso, z_iso, dx_iso, dy_iso, dz_iso, dt = to_isotropic(x, y, z, dx, dy, dz, mass)
    initial_pos.append([0, x_iso, y_iso, z_iso])
    vel.append([dt, dx_iso, dy_iso, dz_iso])
# Some sideways rays along the x-axis
for x0 in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]:
    x, y, z, dx, dy, dz = x0, 0, 0, 0, 0, 1.0
    x_iso, y_iso, z_iso, dx_iso, dy_iso, dz_iso, dt = to_isotropic(x, y, z, dx, dy, dz, mass)
    initial_pos.append([0, x_iso, y_iso, z_iso])
    vel.append([dt, dx_iso, dy_iso, dz_iso])

steps = 2000
daffine = 0.01
initial_pos = np.array(initial_pos).T
pos = np.empty((steps + 1, 4, initial_pos.shape[1]))
pos[0] = initial_pos.copy()
vel = np.array(vel).T
for step in range(steps):
    pos[step+1], vel = geodesic_leapfrog_step(pos[step], vel, daffine, mass)

x_iso = pos[:,1]
y_iso = pos[:,2]
z_iso = pos[:,3]
r_iso = np.sqrt(x_iso ** 2 + y_iso ** 2 + z_iso ** 2)
x = x_iso * (1 + mass / 2 / r_iso) ** 2
y = y_iso * (1 + mass / 2 / r_iso) ** 2
z = z_iso * (1 + mass / 2 / r_iso) ** 2

plt.figure()
gridspec = plt.gcf().add_gridspec(3, 5)
plt.gcf().add_subplot(gridspec[:3,:3])
plt.gca().add_patch(plt.Circle((0, 0), 2 * mass, color="black", fill=False))
plt.gca().add_patch(plt.Circle((0, 0), 3 * mass, color="black", linestyle="--", fill=False))
for i in range(pos.shape[2]):
    plt.plot(x, z)
plt.gca().set_aspect("equal", adjustable="box")
plt.gcf().add_subplot(gridspec[0,3:])
for i in range(pos.shape[2]):
    plt.plot(x)
plt.gcf().add_subplot(gridspec[1,3:])
for i in range(pos.shape[2]):
    plt.plot(y)
plt.gcf().add_subplot(gridspec[2,3:])
for i in range(pos.shape[2]):
    plt.plot(z)
plt.gcf().set_tight_layout(True)
plt.show()