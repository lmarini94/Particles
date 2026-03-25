# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 18:09:22 2026

@author: marin
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from load_metadata import select_run, load_data


###############################################################################
################################# LOAD DATA ###################################
###############################################################################

myrun = select_run()
metadata, states, energies = load_data(myrun)
    

#PHYSICAL PARAMETER
L = metadata["parameters"]["physical"]["L"]
N = metadata["parameters"]["physical"]["N"]
r_c = metadata["parameters"]["physical"]["r_c"]
K_0 = metadata["parameters"]["physical"]["K_0"] 
delta = metadata["parameters"]["physical"]["delta"]             
k_wall = metadata["parameters"]["physical"]["k_wall"]

#SIMULATION PARAMETER
speed = metadata["parameters"]["simulation"]["speed"]
h = metadata["parameters"]["simulation"]["h"]
T_MAX = metadata["parameters"]["simulation"]["T_MAX"]


#MODEL TIMING
created_at = metadata["Created_at"]
t_tot = metadata["Simulation_duration"]
formatted_time = time.strftime("%H:%M:%S", time.gmtime(t_tot))

print("\nSimulated on " + created_at)
print("\n-------Physical parameters-------")
print("Number of particles N =", N)
print("L =", L)
print("Cutoff radius =", r_c)
print("Initial energy K0 =", K_0)
print("Size of wall buffer =", delta)
print("k_wall =", k_wall)
print("------Simulation parameters------")
print("Speed =", speed)
print("Total time =", T_MAX)
print("Step of the simulation h =", h)
print("-------------Timing--------------")
print(f"Simulation completed in {formatted_time}")



###############################################################################
########################## ANALYSIS AND DIAGNOSTICS ###########################
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


def radial_distribution(states, L, r_cut, bins=100, r_max=None):
    """
    Input:
        states = (T, N, 2) dimensional array with the positions of the system
        L = size of the box
        bins = number of radial bins
        r_max = maximum radius considered, default L/2.
    Output:
        r = array of len(bins) of equally spaced radii between 0 and r_max
        g = array of len(bins), values of the RDF
    """
    T, N, _ = states.shape
    
    if r_max is None:
        r_max = min(L/2, r_cut)

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

def rdf_time_series(states, physical_time, L, r_cut, window = 20, step = 10, bins = 100, r_max = None):
    """
    Input:
        states = (T, N, 2) dimensional array with the positions of the system
        L = size of the box
        window = size of temporal window used for time resolved RDF
        step = how often we compute the RDF of the window
        bins = number of radial bins
        r_max = maximum radius considered, default L/2.
    Output:
        times_idx = array of the start time of each window.
        r = array of len(bins) of equally spaced radii between 0 and r_max
        g_t = array of size (len(times_idx), bin), g_t[i] are the values of the 
        RDF function computed between times_idx[i]:times_idx[i] + window
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
    r, _ = radial_distribution(states[:1], L, r_c, bins = bins, r_max = r_max,)
    
    #Compute the array of times 
    times_idx = np.arange(0, T - window + 1, step)
    times_phys = physical_time[times_idx + window // 2]
    g_t = np.empty((len(times_idx), bins), dtype = float) 
    
    for i, t0 in enumerate(times_idx):
        _, g = radial_distribution(states[t0:(t0+window)], L, r_c, bins=bins, r_max = r_max)
        g_t[i] = g
    
    return times_phys, r, g_t


r, g = radial_distribution(states, L, r_c)

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
times_phys, r, g_t = rdf_time_series(states, physical_time, L, r_c)
plt.figure()
plt.imshow(
    g_t,
    aspect="auto",
    origin="lower",
    extent=[r[0], r[-1], times_phys[0], times_phys[-1]]
)
plt.colorbar(label="g(r, t)")
plt.xlabel("r")
plt.ylabel("Physical time (window midpoint)")
plt.title("Time-resolved radial distribution function")
plt.savefig(myrun/"time_resolved_RDF.jpg")
plt.show()
