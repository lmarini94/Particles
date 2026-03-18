# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:19:28 2026

@author: marin
"""

import numpy as np
import time, json
from pathlib import Path
from loading import load_config


###############################################################################
############################### MAIN FUNCTIONS ################################
###############################################################################

def initialize_pos(p, rng, max_tries = 2000):  
    """
    Input:
        p = Instance of SimParameters
        rng = random number generator
        max_tries = maximum number of attempts
    Output:
        pos = (N, 2) dimensional numpy array of positions
    This function generates N points randomly in [delta, L-delta] so that their 
    mutual distance is greater than d_min. Points are generated using inputed rng.    
    At every iteration a new point is generated ensuring that its dstance from
    all previously generated points is d_min. If not, point is rejected and new 
    attempt is made at most max_tries times. 
    """
    # Initialize empty array
    
    pos = np.empty((p.N, 2), dtype=float)
    
    placed = 0
    tries = 0
    
    d2_min = p.d_min * p.d_min

    while placed < p.N:
        if tries > max_tries:
            raise RuntimeError("Failed to place all particles. Decrease d_min or increase L.")
        
        #Candidate [x, y] where x and y are generated uniformly on [delta, L-delta] 
        candidate = rng.random(2) * (p.L-2*p.delta) + p.delta
        
        #First particle is placed
        if placed == 0:
            pos[0] = candidate
            placed = 1
            continue
        tries += 1
        
        #Distance of candidate from previously placed particles, size (placed,2)
        d = pos[:placed] - candidate
        d2 = np.sum(d*d, axis=1)
        #d2 has size (placed), d2[i] = squared distance of particle i and candidate
        
        if np.all(d2 >= d2_min):
            pos[placed] = candidate
            placed += 1
            tries = 0
    
    return pos

def initialize_vel(p, rng):
    """
    Input: 
        p = Instance of SimParameters
        rng = input random number generator
    Output:
        v = (N,2) array of velocities
    This function generates the velocity vectors of N particles. Velocities are 
    generated according to a normal distribution (given rng), net momentum is 
    removed and they are rescaled so that the system has a target kinetic energy.
    """
    
    #V generated accoding to (0, 1) normal. v is shape (N, 2)
    v = rng.normal(0.0, 1.0, (p.N, 2))

    # remove net momentum
    v -= v.mean(axis = 0)

    #Compute total kinetic energy
    K = 0.5*np.sum(v**2)

    scale = np.sqrt(p.K_0 / K)
    v *= scale

    return v

def build_cell(x, n_cells, cell_size):
    """
    Input:
        x = (N,2) np.array of positions.
        n_cells = number of cells per side
        cell_size = side of the square cells
    Output:
        cells = n_cells^2 len array of np.arrays where cells[c] is the array which 
        contains the indices of particles in cell c
    Given a configuration of the system x, this function builds the cell lists.
    """
    
    cells = [[] for _ in range(n_cells**2)]
    
    for i, p in enumerate(x):
        #Vectorial coordinates of cell
        cx = int(p[0]/cell_size)
        cy = int(p[1]/cell_size)
        #Scalar coordinates of cell
        c = cx + cy*n_cells
        cells[c].append(i)
    
    #Convert into np.arrays
    cells = [np.array(c, dtype = int) for c in cells]
    return cells




    

def step (x, v, cells, r_c, L, delta, k_wall, eps, h):
    """
    Input:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
        cells = array of np.arrays where cells[c] is the array which 
        contains the indices of particles in cell c
        r_c = cutoff radius
        L = size of box
        delta = size of wall buffer
        k_wall = wall elastic constant
        eps = regularization parameter
        h = step of integrator.
    Output:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
    At every iteration, this function updates the positions and velocities
    using a velocity-Vertlet symplectic integrator. 
    """
    
    f, _ = forces_potential_interactions(x, cells, r_c, eps)
    f_wall, _ = forces_potential_wall(x, L, delta, k_wall)
    f += f_wall
    
    v += 0.5*h*f
    
    x += h*v
    
    f, _ = forces_potential_interactions(x, cells, r_c, eps)
    f_wall, _ = forces_potential_wall(x, L, delta, k_wall)
    f += f_wall
    
    v += 0.5*h*f
    
    return x, v

###############################################################################
############################### MAIN EXECUTION ################################
###############################################################################



INITIALIZATION OF VARIABLES

start_time = time.time()
states = []
energies = []

x = initialize_pos(N, L, d_min, delta, rng)
v = initialize_vel(N, K_0, rng)

n_cells = int(L/r_c)
cell_size = L/n_cells
r_skin = cell_size - r_c

cells = build_cell(x, n_cells, cell_size)


t_sim = 0 
t = 0
x_ref = x.copy()

while (t_sim <= T_MAX):
    if t%speed == 0:
        states.append(x.copy())
        K, U_inter, U_wall, E_tot = energy(x, v, cells, r_c, L, delta, k_wall, eps)
        energies.append([t_sim, K, U_inter, U_wall, E_tot])
        print(f"Progress = {(t_sim/T_MAX)*100:.2f}%", end="\r", flush = True)
    x, v = step(x, v, cells, r_c, L, delta, k_wall, eps, h)
    if np.max(np.sum((x - x_ref)**2, axis=1)) > (r_skin/2)**2:
        cells = build_cell(x, n_cells, cell_size)
        x_ref = x.copy()
    t_sim = t*h
    t += 1
print(f"Progress = {(t_sim/T_MAX)*100:.2f}%", end="\r", flush = True)
total_time = time.time()-start_time
states = np.array(states)
energies = np.array(energies) 

formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
print(f"\nSimulation completed in {formatted_time}")


###############################################################################
########################## DATA AND METADATA SAVE #############################
###############################################################################

state_path = config["outputs"]["states_file"]
energies_path = config["outputs"]["energies_file"]

# CREATE TIMESTAMPED OUTPUT DIRECTORY
timestamp = time.strftime("%Y%m%d_%H%M")
run_dir = Path("runs") / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

# ADD TIMING PARAMETERS TO CONFIG AND SAVE TO FOLDER
config["Created_at_unix"] = time.time()
config["Created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
config["Simulation_duration"] = total_time
Path(run_dir/"run_parameters.json").write_text(json.dumps(config, indent=2))


# SAVE DATA FILES
np.save(run_dir / state_path, states) 
np.save(run_dir / energies_path, energies)

print(f"Results saved to: {run_dir}")

