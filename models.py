# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:53:08 2026

@author: Ludovico Marini 
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class SimState:
    x: np.ndarray
    v: np.ndarray
    cells: list
    t : int = 0
    t_sim : float = 0.0
    
@dataclass 
class SimParameters:
    N: int
    L: float
    d_min: float
    r_c: float
    K_0: float
    delta: float
    k_wall: float
    n_cells: int
    cell_size: float
    h: float
    eps: float
    speed: int
    T_MAX: float
    seed: int | None

@dataclass
class SimIO:
    states_file: str
    energies_file: str
    
p = SimState()