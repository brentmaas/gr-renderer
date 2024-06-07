import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def normalise_cartesian(x, y, z):
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return x / d, y / d, z / d

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
    
    return npos, nvel, inside_horizon

def disk_transparency_at(r, disk_min, disk_max):
    return 1 / r / (1 / disk_min - 1 / disk_max) + 1 / (1 - disk_max / disk_min)

def colour_step(col, pos, npos, mass, disk_min, disk_max):
    nr_iso = np.sqrt(npos[1] * npos[1] + npos[2] * npos[2] + npos[3] * npos[3])
    nr = nr_iso * (1 + mass / 2 / nr_iso) ** 2
    disk_intersect = np.logical_and.reduce((npos[3] * pos[3] < 0, nr >= disk_min, nr <= disk_max))
    transparency = disk_transparency_at(nr[disk_intersect], disk_min, disk_max)
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
x_iso, y_iso, z_iso, dx_iso, dy_iso, dz_iso, dt = to_isotropic(start_pos[0], start_pos[1], start_pos[2], dx, dy, dz, mass)

steps = 1000
daffine = -0.05
pos = np.empty((steps + 1, 4, size[0] * size[1]))
pos[0,0] = 0
pos[0,1] = x_iso
pos[0,2] = y_iso
pos[0,3] = z_iso
vel = np.array([dt, dx_iso, dy_iso, dz_iso])
col = np.zeros((4, size[0] * size[1]))
col[3] = 1
inside_horizon = np.zeros(size[0] * size[1], dtype=bool)
for step in tqdm(range(steps)):
    pos[step+1], vel, inside_horizon = geodesic_leapfrog_step(pos[step], vel, daffine, mass)
    col = colour_step(col, pos[step], pos[step+1], mass, disk_min, disk_max)
col = colour_finalise(col, inside_horizon)

img = np.zeros((size[1], size[0], 3))
for i in range(3):
    img[:,:,i] = col[i].reshape((size[1], size[0]))
plt.figure()
plt.imshow(img)
plt.gcf().set_tight_layout(True)
plt.show()