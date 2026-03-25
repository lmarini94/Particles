# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 18:09:22 2026

@author: marin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import time
from load_metadata import select_run, load_data

###############################################################################
################################# LOAD DATA ###################################
###############################################################################

myrun = select_run()
metadata, states, energies = load_data(myrun)

#Recall energies is of the type [t_sim, K, U, U_wall, E_tot]
physical_time = energies[:, 0]

#PHYSICAL PARAMETER
L = metadata["parameters"]["physical"]["L"]
N = metadata["parameters"]["physical"]["N"]
K_0 = metadata["parameters"]["physical"]["K_0"] 

#SIMULATION PARAMETER
T_MAX = metadata["parameters"]["simulation"]["T_MAX"]

#MODEL TIMING
created_at = metadata["Created_at"]
t_tot = metadata["Simulation_duration"]
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
    f"$K_0$ = {K_0}\n"
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

