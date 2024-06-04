import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def normalise_cartesian(x, y, z):
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return x / d, y / d, z / d

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

def geodesic_leapfrog_step(pos, vel, daffine, mass, with_inside_horizon=False):
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
    
    if with_inside_horizon:
        return npos, nvel, inside_horizon
    return npos, nvel

def disk_transparency_at(r):
    return 1 / r / (1 / disk_min - 1 / disk_max) + 1 / (1 - disk_max / disk_min)

def colour_step(col, pos, npos):
    disk_intersect = np.logical_and.reduce(((npos[2] - np.pi / 2) * (pos[2] - np.pi / 2) < 0, npos[1] >= disk_min, npos[1] <= disk_max))
    transparency = disk_transparency_at(npos[1,disk_intersect])
    for i in range(3):
        col[i,disk_intersect] += disk_colour[i] * transparency * col[3,disk_intersect]
    col[3,disk_intersect] -= transparency * col[3,disk_intersect]
    
    return col

def colour_finalise(col, inside_horizon):
    for i in range(3):
        col[i,~inside_horizon] += sky_colour[i] * col[3,~inside_horizon]
    col[3] = 0
    
    return col

mass = 1
disk_min = 8
disk_max = 16
disk_colour = (1, 0.5, 0)
sky_colour = (0, 0, 0.2)
size = (1000, 600)
hfov = 90 * np.pi / 180
vfov = hfov * size[1] / size[0]
start_r = 20
start_theta = np.pi / 2 - np.pi / 12
start_phi = 0
start_pos = (start_r * np.sin(start_theta), 0, start_r * np.cos(start_theta))

j, i = np.mgrid[:size[1],:size[0]]
dx = np.ones(size[0] * size[1])
dy = np.sin(2 * (i.flatten() / size[0] - 0.5) * hfov)
dz = np.sin(2 * (j.flatten() / size[1] - 0.5) * vfov)
dx, dy, dz = normalise_cartesian(dx, dy, dz)
dx, dz = dx * np.sin(start_theta) - dz * np.cos(start_theta), dz * np.sin(start_theta) + dx * np.cos(start_theta)
dx, dy = dx * np.cos(start_phi) + dy * np.sin(start_phi), dy * np.cos(start_phi) - dx * np.sin(start_phi)
initial_r, initial_theta, initial_phi, initial_dr, initial_dtheta, initial_dphi = pos_vel_to_polar(start_pos[0], start_pos[1], start_pos[2], dx, dy, dz)
initial_dt = np.sqrt(initial_dr ** 2 / (1 - 2 * mass / initial_r) ** 2 + initial_r ** 2 / (1 - 2 * mass / initial_r) * initial_dtheta ** 2 + initial_r ** 2 * np.sin(initial_theta) ** 2 / (1 - 2 * mass / initial_r) * initial_dphi ** 2)

steps = 1000
daffine = -0.05
pos = np.empty((steps + 1, 4, size[0] * size[1]))
pos[0,0] = 0
pos[0,1] = initial_r
pos[0,2] = initial_theta
pos[0,3] = initial_phi
vel = np.array([initial_dt, initial_dr, initial_dtheta, initial_dphi])
col = np.zeros((4, size[0] * size[1]))
col[3] = 1
inside_horizon = np.zeros(size[0] * size[1], dtype=bool)
for step in tqdm(range(steps)):
    pos[step+1], vel, inside_horizon = geodesic_leapfrog_step(pos[step], vel, daffine, mass, with_inside_horizon=True)
    col = colour_step(col, pos[step], pos[step+1])
col = colour_finalise(col, inside_horizon)

x, y, z = pos_to_cartesian(pos[:,1], pos[:,2], pos[:,3])
img = np.zeros((size[1], size[0], 3))
for i in range(3):
    img[:,:,i] = col[i].reshape((size[1], size[0]))
plt.figure()
plt.imshow(img)
plt.gcf().set_tight_layout(True)
plt.show()