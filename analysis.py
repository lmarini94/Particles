# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 18:09:22 2026

@author: marin
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, time

###############################################################################
############################# SELECTION OF RUN ################################
###############################################################################


def select_run(latest=True, runs_dir="run"):
    runs_dir = Path(runs_dir)
    
    #Check if run directory exists
    if not runs_dir.exists():
        raise FileNotFoundError(f"Run directory '{runs_dir}' does not exist.")

    runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])

    if not runs:
        raise RuntimeError("No runs found in directory.")

    if latest:
        run_name = runs[-1]
        print(f"Using latest run: {run_name}")
        return runs_dir / run_name

    # manual selection
    run_name = input("Enter run timestamp (YYYYMMDD_HHMM): ")

    if run_name not in runs:
        raise ValueError(f"Run '{run_name}' not found in {runs_dir}")
    
    print(f"Using run: {run_name}")
    return runs_dir / run_name

myrun = select_run()

###############################################################################
############################### IMPORT DATA ###################################
###############################################################################

# LOAD CONFIGURATION FILE
CONFIG_PATH = Path(myrun/"run_parameters.json")
with open(CONFIG_PATH) as f:
    config = json.load(f)
    
#LOAD STATES AND ENERGIES FILE
states = np.load(myrun/"states.npy")
energies = np.load(myrun/"energies.npy")

#PHYSICAL PARAMETER
L = config["physical"]["L"]
N = config["physical"]["N"]
K_target = config["physical"]["K_target"] 
delta = config["physical"]["delta"]             
k_wall = config["physical"]["k_wall"]

#SIMULATION PARAMETER
speed = config["simulation"]["speed"]
h = config["simulation"]["h"]
T_MAX = config["simulation"]["T_MAX"]


#MODEL TIMING
created_at = config["Created_at"]
t_tot = config["Simulation_duration"]
formatted_time = time.strftime("%H:%M:%S", time.gmtime(t_tot))

print("\nSimulated on " + created_at)
print("\n-------Physical parameters-------")
print("Number of particles N =", N)
print("L =", L)
print("Initial energy K0 =", K_target)
print("Size of wall buffer =", delta)
print("k_wall =", k_wall)
print("------Simulation parameters------")
print("Speed =", speed)
print("Total time =", T_MAX)
print("Step of the simulation h =", h)
print("-------------Timing--------------")
print(f"Simulation completed in {formatted_time}")



###############################################################################
########################## ANALYSIS AND DIAGNOSTICS ############################
###############################################################################

#Recall energies is of the type [t_sim, K, U, U_wall, E_tot]
physical_time = energies[:, 0]  #Physical time
K = energies[:, 1]              #Kinetic energy
U_int = energies[:, 2]          #Interaction potential
U_wall = energies[:, 3]         #Wall potential
E = energies[:, 4]              #Total energy

dE = E -E[0]                    #Energy drift
dE_rel = dE/abs(E[0])           #Relative energy drift

print("-------Stability of energy-------")
print("E0 =", E[0])
print("U_int =", U_int[0])
print("U_wall0 =", U_wall[0])
print("K0 =", K[0])
print("rel max dev =", np.max(np.abs(dE)) / abs(E[0]))
print("rel range =", (E.max() - E.min()) / abs(E[0]))


###############################################################################
############################ RADIAL DISTRIBUTION ##############################
###############################################################################


def radial_distribution(states, L, bins=100, r_max=None):
    """
    This function computes the radial distribution function. 
    
    states = (T, N, 2) array with x, y per frame. 
    L = side of the box.
    bins = number of radial bins
    r_max = max radius to consider, default = L/2
    """
    T, N, _ = states.shape
    
    if r_max is None:
        r_max = L/2

    # particle density
    rho = N / (L*L)

    # histogram bins
    edges = np.linspace(0, r_max, bins+1)
    dr = r_max/bins
    r = 0.5*(edges[:-1] + edges[1:])

    g = np.zeros(bins)
    
    iu, ju = np.triu_indices(N, k=1)

    for frame in states:

        x = frame[:, 0]
        y = frame[:, 1]

        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]

        dist = np.sqrt(dx*dx + dy*dy)

        hist, _ = np.histogram(dist, edges)
        g += hist

    # average over frames
    g /= T

    # normalization
    shell_area = 2*np.pi*r*dr
    ideal = rho * shell_area * N / 2.0

    g = g / ideal

    return r, g

def rdf_time_series(states, L, window = 20, step = 10, bins = 100, r_max = None):
    """
    Compute time-resolved radial distribution function g_t(r).

    states: array (T, N, 2) with x,y per frame
    L: box size (assumes [0,L]x[0,L])
    bins: number of radial bins
    r_max: max radius to consider (default L/2)
    window: number of consecutive frames to average (1 = per-frame RDF)
    step: stride between output times

    Returns:
      times_idx: indices of the 'time points' (start of each window)
      r: bin centers (array of size(r) = bin)
      g_t: array (size(times_idx), bins) with g_t(r) over time
    """
    
    T = len(states)
    
    if window < 1 or window > T:
        raise ValueError("window must be between 1 and T")
    if step < 1:
        raise ValueError("step must be >= 1")
    if not isinstance(step, int):
        raise TypeError("Step must be integer")
    if not isinstance(window, int):
        raise TypeError("Window must be integer")
        

    
    #Computes r once
    r, _ = radial_distribution(states[:1], L, bins = bins, r_max = r_max,)
    
    #Compute the array of times 
    times_idx = np.arange(0, T - window + 1, step)
    g_t = np.empty((len(times_idx), bins), dtype = float) 
    
    for i, t0 in enumerate(times_idx):
        _, g = radial_distribution(states[t0:(t0+window)], L, bins=bins, r_max = r_max)
        g_t[i] = g
    
    return times_idx, r, g_t


r, g = radial_distribution(states, L)

###############################################################################
################################### GRAPHS ####################################
###############################################################################


#RELATIVE ENERGY DRIFT
plt.plot(physical_time, dE_rel, linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=0.5)
plt.xlabel("Physical time")
plt.ylabel("Relative energy drift $(E-E_0)/|E_0|$")
plt.title("Energy drift")
plt.savefig(myrun/"energy_drift.jpg", bbox_inches="tight", dpi=200)
plt.show()

#POTENTIAL, WALL AND KINETIC ENERGY
plt.plot(physical_time, U_wall/abs(E[0]), label = "Wall potential", linewidth = 0.5, color = "red")
plt.plot(physical_time, U_int/abs(E[0]), label = "Interaction potential",linewidth = 0.5, color = "blue")
plt.plot(physical_time, K/abs(E[0]), label = "Kinetic", linewidth = 0.5, color = "green")
plt.xlabel("Physical time")
plt.ylabel("Normalized energies")
plt.title("Energy components vs time")
plt.legend()
plt.tight_layout()
plt.savefig(myrun/"energy_components.jpg")
plt.show()

#STATIC RADIAL DISTRIBUTION FUNCTION
plt.figure()
plt.plot(r, g, lw=0.5)
plt.axhline(1, color = "red", linestyle="--", lw=0.5)
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial distribution function")
plt.savefig(myrun/"RDF.jpg")
plt.show()

#TIME RESOLVED RADIAL DISTRIBUTION FUNCTION
times_idx, r, g_t = rdf_time_series(states, L)
plt.figure()
plt.imshow(
    g_t,
    aspect="auto",
    origin="lower",
    extent=[r[0], r[-1], times_idx[0], times_idx[-1]]
)
plt.colorbar(label="g(r, t)")
plt.xlabel("r")
plt.ylabel("frame index (window start)")
plt.title("Time-resolved radial distribution function")
plt.savefig(myrun/"time_resolved_RDF.jpg")
plt.show()
