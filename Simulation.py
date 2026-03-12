# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:19:28 2026

@author: marin
"""

import numpy as np
import time, json
from pathlib import Path


###############################################################################
################################# PARAMETERS ##################################
###############################################################################

# LOAD CONFIGURATION FILE
CONFIG_PATH = Path("CONFIG.json")
with open(CONFIG_PATH) as f:
    config = json.load(f)
    
#EXTRACT PARAMETERS

#PHYSICAL PARAMETERS
#N -> Number of particles
#L -> Size of the box
#d_min -> Min initial dist between particles
#r_c -> cutoff radius
#K_target -> Initial kinetik energy
#delta -> Size of buffer zone around walls
#k_wall -> Elastic constant of wall

N = config["physical"]["N"]                     
L = config["physical"]["L"]                     
d_min = config["physical"]["d_min"] 
r_c = config["physical"]["r_c"]            
K_target = config["physical"]["K_target"]       
delta = config["physical"]["delta"]             
k_wall = config["physical"]["k_wall"]

#SIMULATION PARAMETERS
#h -> Pace of the simulation
#eps -> Smoothing parameter for close range interactions
#speed -> Speed at which we store positions/energies
#T_MAX -> Total time of the simulation

h = config["simulation"]["h"]
eps = config["simulation"]["eps"]
speed = config["simulation"]["speed"]
T_MAX = config["simulation"]["T_MAX"]
seed = config["simulation"]["seed"]


rng = np.random.default_rng(seed)
#If "seed":null then seed = None


###############################################################################
############################### MAIN FUNCTIONS ################################
###############################################################################

def initialize_pos(N, L, d_min, delta, rng, max_tries = 2000):  
    """
    Input:
        N = number of particles
        d_min = minimum distance between particles
        delta = size of wall buffer zone
        rng = random number generator
        max_tries = maximum number of attempts
    Output:
        pos = (N, 2) dimensional array of positions
    This function generates N points randomly in [delta, L-delta] so that their 
    mutual distance is greater than d_min. Points are generated using inputed rng.    
    At every iteration a new point is generated ensuring that its dstance from
    all previously generated points is d_min. If not, point is rejected and new 
    attempt is made at most max_tries times. 
    """
    
    # Initialize empty array
    pos = np.empty((N, 2), dtype=float)
    
    placed = 0
    tries = 0
    
    d2_min = d_min * d_min

    while placed < N:
        if tries > max_tries:
            raise RuntimeError("Failed to place all particles. Decrease d_min or increase L.")
        
        #Candidate [x, y] where x and y are generated uniformly on [delta, L-delta] 
        candidate = rng.random(2) * (L-2*delta) + delta
        
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

def initialize_vel(N, K_target, rng):
    """
    Input: 
        N = number of particles
        K_target = target kinetik energy of the system of particles
        rng = input random number generator
    Output:
        v = (N,2) array of velocities
    This function generates the velocity vectors of N particles. Velocities are 
    generated according to a normal distribution (given rng), net momentum is 
    removed and they are rescaled so that the system has a target kinetik energy.
    """
    
    #V generated accoding to (0, 1) normal. v is shape (N, 2)
    v = rng.normal(0.0, 1.0, (N, 2))

    # remove net momentum
    v -= v.mean(axis = 0)

    #Compute total kinetik energy
    K = 0.5*np.sum(v**2)

    scale = np.sqrt(K_target / K)
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

def forces_potential_interactions(x, cells, r_c, eps):
    """
    Input: 
        x = (N,2) np.array of positions.
        cells = array of np.arrays where cells[c] is the array which 
        contains the indices of particles in cell c
        r_c = cutoff radious
        eps = regularization parameter
    Output: 
        f = (N, 2) array of forces
        U = potential energy due to interactions
    This function computes the array of forces f and the potential energy U
    of particle interactions using a cell list approach. Forces are computed 
    using the potential U(r) = 1/r^6 -1/r^4 which is set to zero for r>r_c and
    shifted to ensure continuity. U = U(r) -U(r_c). A smoothing correction 
    of eps is added to the computation of the distance between particles.
    """
    N = len(x)
    n_cells = int(np.sqrt(len(cells)))
    rc2 = r_c*r_c
    eps2 = eps*eps
    #Compute potential shift
    U_shift = 1/(r_c)**6 - 1/(r_c)**4
    
    f = np.zeros((N, 2))
    U = 0.0
    
    #Neighbouring cells that will be checked 
    neighbors = [(0,0),(1,0),(-1,1),(0,1),(1,1)]
    
    #Cycle across all cells
    for cy in range(n_cells):
        for cx in range(n_cells):
            #Cycle across all cells
            c = cx + cy*n_cells
            A = cells[c]
            
            #If cell is empty skip
            if len(A)==0:
                continue
            
            #Check all 5 neighbouring cells
            for nx, ny in neighbors:
                #If neighbour goes out of box skip
                if (cx + nx) < 0 or (cx + nx) >= n_cells or (cy + ny) >= n_cells:
                    continue
                n = cx + nx + (cy + ny)*n_cells
                B = cells[n]
                
                #If neighbouring cell is empty skip
                if len(B) == 0:
                    continue
                
                #Compute differences this is shape len(A), len(B), 2
                #Here dx[a, b] = [dx, dy] where dx and dy are the distances between
                #particle x[A[a]] and x[B[b]]
                dx = x[A][:, None, :] - x[B][None, :, :]
                
                #r2 is shape len(A), len(B)
                r2 = np.sum(dx*dx, axis = 2)
                
                mask = r2 < rc2
                
                #If all particles distant more than the cutoff, skip
                if not np.any(mask):
                    continue
                
                i_idx, j_idx = np.where(mask)
                
                #If same cell, only use upper triangle
                if c == n:
                    valid = i_idx < j_idx
                    i_idx = i_idx[valid]
                    j_idx = j_idx[valid]
                
                pair_r2 = r2[i_idx, j_idx]
                pair_dx = dx[i_idx, j_idx]
                
                inv_r2 = 1.0 / (pair_r2 + eps2)
                inv_r4 = inv_r2 * inv_r2
                inv_r6 = inv_r4 * inv_r2
                inv_r8 = inv_r4 * inv_r4
                
                coef = 6*inv_r8 - 4*inv_r6
                
                
                np.add.at(f, A[i_idx],  coef[:, None] * pair_dx)
                np.add.at(f, B[j_idx], -coef[:, None] * pair_dx)
                
                U += np.sum(inv_r6 - inv_r4 - U_shift)
    return f, U


def forces_potential_wall(x, L, delta, k_wall):
    """
    Input:
        x = (N, 2) dimensional array of positions
        delta = size of wall buffer
        k_wall = wall elastic constant
    Output:
        f = (N, 2) array of forces
        U = elastic potential energy
    This function, computes the forces due to the wall interactions and the 
    corresponding potetial energy.This is modelled as an elastic force which 
    acts on a buffer zone of lenght delta from the walls.
    """
    f = np.zeros_like(x)
    U = 0.0
    
    #Upper wall x = L, y= L
    active = x > (L-delta)
    d = x[active] - (L - delta)
    f [active] += -k_wall*d
    U += 0.5 * k_wall * np.sum(d*d)
    
    #Lower wall x = 0, y= 0
    active = x < delta
    d = delta - x[active]
    f [active] += k_wall*d
    U += 0.5 * k_wall * np.sum(d*d)
    
    return f, U

def energy(x, v, cells, r_c, L, delta, k_wall, eps):
    """
    Input:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
        cells = array of np.arrays where cells[c] is the array which 
        contains the indices of particles in cell c
        r_c = cutoff radious
        L = size of box
        delta = size of wall buffer
        k_wall = wall elastic constant
        eps = regularization parameter
    Output :
        K = kinetik energy 
        U_inter = potential energy due to interparticle forces
        U_wall = potential due to wall interactions
        E_tot = total energy 
    This function, given a state of the system x, v (positions and 
    velocities) returns the various energies of the system.
    """
    #K = 1/2 v^2
    K = 0.5 * np.sum(v**2)
    _, U_inter = forces_potential_interactions(x, cells, r_c, eps)
    _, U_wall = forces_potential_wall(x, L, delta, k_wall)
    
    E_tot = K + U_inter + U_wall
    return K, U_inter, U_wall, E_tot

def dmin(x):
    """
    Input:
        x = (N, 2) dimensional array of positions
    Output:
        r_min = the minimum distance between particles 
    """
    N = len(x)
    iu, ju = np.triu_indices(N, k=1)
    
    d = x[iu] - x[ju]
    
    r2 = np.sum(d*d, axis=1)
    r_min = np.sqrt(np.min(r2))
    return r_min
    

def step (x, v, cells, r_c, L, delta, k_wall, eps, h):
    """
    Input:
        x = (N, 2) dimensional position vector
        v = (N, 2) dimentsional velocity vector
        cells = array of np.arrays where cells[c] is the array which 
        contains the indices of particles in cell c
        r_c = cutoff radious
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

# INITIALIZATION OF VARIABLES

start_time = time.time()
states = []
energies = []

x = initialize_pos(N, L, d_min, delta, rng)
v = initialize_vel(N, K_target, rng)

n_cells = int(L/r_c)
cell_size = L/n_cells

cells = build_cell(x, n_cells, cell_size)


t_sim = 0 
t = 0

while (t_sim <= T_MAX):
    if t%speed == 0:
        states.append(x.copy())
        K, U_inter, U_wall, E_tot = energy(x, v, cells, r_c, L, delta, k_wall, eps)
        energies.append([t_sim, K, U_inter, U_wall, E_tot])
        print(f"Progress = {(t_sim/T_MAX)*100:.2f}%", end="\r", flush = True)
    x, v = step(x, v, cells, r_c, L, delta, k_wall, eps, h)
    cells = build_cell(x, n_cells, cell_size)
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

