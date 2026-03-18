# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:05:57 2026

@author: marin
"""

from loading import load_config
from core import Simulation

def main():
    params, paths, rng = load_config()
    sim = Simulation(params, rng)
    sim.initialize()
    sim.run()
    sim.save(paths) 

if __name__ == "__main__":
    main()