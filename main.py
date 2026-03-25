# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:05:57 2026

@author: marin
"""

from simulation.loading import load_config
from simulation.core import Simulation

def main():
    #Load simulation parameters, path
    params, outputs, rng = load_config()
    sim = Simulation(params, rng)
    sim.initialize()
    sim.run()
    sim.save(outputs) 

if __name__ == "__main__":
    main()