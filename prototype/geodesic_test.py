import numpy as np
import matplotlib.pyplot as plt

def pos_vel_to_polar(x, y, z, dx, dy, dz):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    
    v = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    vtheta = np.arccos(dz / v)
    vphi = np.arctan2(dy, dx)
    
    dr = np.sin(vtheta) * np.cos(vphi) * np.sin(theta) * np.cos(phi) + np.sin(vtheta) * np.sin(vphi) * np.sin(theta) * np.sin(phi) + np.cos(vtheta) * np.cos(theta)
    dtheta = (np.sin(vtheta) * np.cos(vphi) * np.cos(theta) * np.cos(phi) + np.sin(vtheta) * np.sin(vphi) * np.cos(theta) * np.sin(phi) - np.cos(vtheta) * np.sin(theta)) / r
    dphi = (-np.sin(vtheta) * np.cos(vphi) * np.sin(phi) + np.sin(vtheta) * np.sin(vphi) * np.cos(phi)) / (r * np.sin(theta))
    
    return r, theta, phi, dr, dtheta, dphi

def pos_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z

def christoffel_symbols(r, theta, phi, mass):
    """
    Gamma^mu_rho_sigma -> array[mu][rho][sigma]
    mu, rho, sigma -> t, r, theta, phi
    """
    
    zero = np.zeros(len(r)) if type(r) == np.ndarray else 0
    return np.nan_to_num(np.array(
    [
        [ # t
            [zero, mass / r / (r - 2 * mass), zero, zero], # t, t
            [mass / r / (r - 2 * mass), zero, zero, zero], # t, r
            [zero, zero, zero, zero], # t, theta
            [zero, zero, zero, zero] # t, phi
        ],
        [ # r
            [mass * (r - 2 * mass) / r ** 3, zero, zero, zero], # r, t
            [zero, -mass / r / (r - 2 * mass), zero, zero], # r, r
            [zero, zero, 2 * mass - r, zero], # r, theta
            [zero, zero, zero, (2 * mass - r) * np.sin(theta) ** 2] # r, phi
        ],
        [ # theta
            [zero, zero, zero, zero], # theta, t
            [zero, zero, 1 / r, zero], # theta, r
            [zero, 1 / r, zero, zero], # theta, theta
            [zero, zero, zero, -np.sin(theta) * np.cos(theta)] # theta, phi
        ], 
        [ # phi
            [zero, zero, zero, zero], # phi, t
            [zero, zero, zero, 1 / r], # phi, r
            [zero, zero, zero, np.cos(theta) / np.sin(theta)], # phi, theta
            [zero, 1 / r, np.cos(theta) / np.sin(theta), zero] # phi, phi
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
    
    inside_horizon = npos[1] <= 2 * mass
    nvel[:,inside_horizon] = vel[:,inside_horizon]
    npos[:,inside_horizon] = pos[:,inside_horizon]
    
    return npos, nvel

mass = 1
initial_pos = []
vel = []
rays = 12
# Uniformly distributed rays from (3, 0)
for i in range(rays):
    r, theta, phi, dr, dtheta, dphi = pos_vel_to_polar(3, 0, 0, np.cos(2 * np.pi * i / rays), 0, np.sin(2 * np.pi * i / rays))
    dt = np.sqrt(dr ** 2 / (1 - 2 * mass / r) ** 2 + r ** 2 / (1 - 2 * mass / r) * dtheta ** 2 + r ** 2 * np.sin(theta) ** 2 / (1 - 2 * mass / r) * dphi ** 2)
    initial_pos.append([0, r, theta, phi])
    vel.append([dt, dr, dtheta, dphi])
# Some sideways rays along the x-axis
for x in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]:
    r, theta, phi, dr, dtheta, dphi = pos_vel_to_polar(x, 0, 0, 0, 0, 1)
    dt = np.sqrt(dr ** 2 / (1 - 2 * mass / r) ** 2 + r ** 2 / (1 - 2 * mass / r) * dtheta ** 2 + r ** 2 * np.sin(theta) ** 2 / (1 - 2 * mass / r) * dphi ** 2)
    initial_pos.append([0, r, theta, phi])
    vel.append([dt, dr, dtheta, dphi])

steps = 2000
daffine = 0.01
initial_pos = np.array(initial_pos).T
pos = np.empty((steps + 1, 4, initial_pos.shape[1]))
pos[0] = initial_pos.copy()
vel = np.array(vel).T
for step in range(steps):
    pos[step+1], vel = geodesic_leapfrog_step(pos[step], vel, daffine, mass)

x, y, z = pos_to_cartesian(pos[:,1], pos[:,2], pos[:,3])

plt.figure()
gridspec = plt.gcf().add_gridspec(3, 5)
plt.gcf().add_subplot(gridspec[:3,:3])
plt.gca().add_patch(plt.Circle((0, 0), 2, color="black", fill=False))
plt.gca().add_patch(plt.Circle((0, 0), 3, color="black", linestyle="--", fill=False))
for i in range(pos.shape[2]):
    plt.plot(x[:,i], z[:,i])
plt.gca().set_aspect("equal", adjustable="box")
plt.gcf().add_subplot(gridspec[0,3:])
for i in range(pos.shape[2]):
    plt.plot(x[:,i])
plt.gcf().add_subplot(gridspec[1,3:])
for i in range(pos.shape[2]):
    plt.plot(y[:,i])
plt.gcf().add_subplot(gridspec[2,3:])
for i in range(pos.shape[2]):
    plt.plot(z[:,i])
plt.gcf().set_tight_layout(True)
plt.show()