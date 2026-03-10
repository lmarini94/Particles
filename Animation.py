# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 18:09:22 2026

@author: marin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import json, time

###############################################################################
############################# SELECTION OF RUN ################################
###############################################################################


def select_run(latest=True, runs_dir="runs"):
    runs_dir = Path(runs_dir)
    
    #Check if runs directory exists
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory '{runs_dir}' does not exist.")

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

#Recall energies is of the type [t_sim, K, U, U_wall, E_tot]
physical_time = energies[:, 0]

#PHYSICAL PARAMETER
L = config["physical"]["L"]
N = config["physical"]["N"]
K_target = config["physical"]["K_target"] 
#SIMULATION PARAMETER
T_MAX = config["simulation"]["T_MAX"]

#MODEL TIMING
created_at = config["Created_at"]
t_tot = config["Simulation_duration"]
formatted_time = time.strftime("%H:%M:%S", time.gmtime(t_tot))

print("Simulated on " + created_at)


###############################################################################
################################# ANIMATION ###################################
###############################################################################

fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Particle animation", fontsize=12, fontweight='bold')
scat = ax.scatter([], [], s = 8)

#PARAMETER BOX
params = (
    f"$N$ = {N}\n"
    f"$L$ = {L}\n"
    f"$K_0$ = {K_target}\n"
    f"$T_{{max}}$ = {T_MAX}"
)

ax.text(
    1.02, 0.98, params,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top'
)

#TIMING BOX
time_text = ax.text(
    1.02, 0.02, "",
    transform=ax.transAxes,
    fontsize=10,
    va='bottom'
)

def init():
    scat.set_offsets(np.empty((0, 2)))
    time_text.set_text("")
    return scat, time_text

def update(i):
    scat.set_offsets(states[i])
    time_text.set_text(f"t = {physical_time[i]:.2f}")
    return scat, time_text

frame_step = 5
ani = FuncAnimation(
    fig, update,
    frames=range(0, len(states), frame_step),
    init_func= init,
    interval=10,
    blit=True
)

animation_path = Path(myrun/"particles_film.gif")
ani.save(animation_path, writer="pillow")
print(f"Animation saved at: {animation_path}" )

