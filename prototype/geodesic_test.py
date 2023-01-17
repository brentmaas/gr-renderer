import numpy as np
import matplotlib.pyplot as plt

def polar_velocity_norm_2d(pos, vel, mass):
    """
    Normalises a velocity vector in 2D polar coordinates such that it is the velocity vector for a light ray at a given position around a Schwarzschild black hole of a given mass.
    """
    mag = np.sqrt(vel[0] ** 2 / (1 - 2 * mass / pos[0]) ** 2 + pos[0] ** 2 * vel[1] ** 2 / (1 - 2 * mass / pos[0]))
    return (vel[0] / mag, vel[1] / mag)

mass = 1
# Polar coordinates (r, phi)
pos = []
vel = []
# Uniformly distributed rays from (3, 0)
rays = 12
for i in range(rays):
    pos.append((3, 0))
    vel.append(polar_velocity_norm_2d((3, 0), (np.cos(2 * np.pi * i / rays), np.sin(2 * np.pi * i / rays) / 3), mass))

def geodesic_leapfrog_step_2d(pos, vel, dt, mass):
    """
    Performs a leapfrog integration step around a 2D Schwarzschild black hole for a given mass.
    Returns the position and velocity after a leapfrog step using the given position, velocity, and timestep.
    """
    
    # Velocity step
    dphi = np.arctan2(pos[0] * vel[1] * dt, pos[0] + vel[0] * dt)
    npos = (np.sqrt((pos[0] + vel[0] * dt) ** 2 + (pos[0] * vel[1] * dt) ** 2), pos[1] + dphi)
    nvel = (vel[0], pos[0] / npos[0] * vel[1])
    
    # Geometry step
    nvel = polar_velocity_norm_2d(npos, (nvel[0] + (-mass / npos[0] ** 3 * (npos[0] - 2 * mass) + mass / npos[0] / (npos[0] - 2 * mass) * nvel[0] ** 2 + (npos[0] - 2 * mass) * nvel[1] ** 2) * dt, nvel[1] + (-1 / npos[0] * nvel[0] * nvel[1]) * dt), mass)
    
    return npos, nvel

posses = [[p for p in pos]]
for _ in range(20000):
    for i in range(rays):
        pos[i], vel[i] = geodesic_leapfrog_step_2d(pos[i], vel[i], 0.001, mass)
    posses.append([p for p in pos])

paths_x = [[] for _ in range(rays)]
paths_y = [[] for _ in range(rays)]
for i in range(len(posses)):
    for j in range(rays):
        paths_x[j].append(posses[i][j][0] * np.cos(posses[i][j][1]))
        paths_y[j].append(posses[i][j][0] * np.sin(posses[i][j][1]))

plt.figure()
plt.gca().add_patch(plt.Circle((0, 0), 2, color="black", fill=False))
plt.gca().add_patch(plt.Circle((0, 0), 3, color="black", linestyle="--", fill=False))
for i in range(rays):
    plt.plot(paths_x[i], paths_y[i])
plt.show()