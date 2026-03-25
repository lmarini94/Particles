# Particles

This projects implements a **two-dimensional molecular dynamics** simulation of interacting particles confined in a box with soft walls. 
The interaction between particles uses a **Lennard-Jones type potential** while the soft wall is modelled as an elastic force acting within a buffer zone from the wall. 
The numerical simulation is carried out using a **Velocity - Vertlet algorithm** to integrate the equations of motion. 
Tools for energy monitoring and post-simulation analysis are also provided. 

## Model details

- ### Initialization (see `loading.py` and `initialization.py`)
  At the beginning of the simulation $N$, particles are randomly generated inside a square box of side $\[\delta, L-\delta\]$ so that their mutual distance is at least $d_{\min}$.
  The velocities are then randomly generated according to a normal distribution, net momentum is removed and they are rescaled to achieve a target initial kinetic energy.
- ### Soft wall (see `forces.py`)
  Particles are confined in the square box using **soft (penetrable) walls** rather than perfectly reflecting boundaries.
  When a particle gets within $\delta$ of a boundary, it experiences an **elastic force** directed back toward the interior.
  This approach avoids discontinuous velocity flips at the boundary and yields a continuous, energy-based wall interaction.
- ### Potential (see `forces.py`)
  The potential between particles is modelled after $U(r) = \frac{1}{r^6} - \frac{1}{r^4}$.
  To avoid instability at small distances, this potential is regularized substituting $r^2 \rightarrow r^2 + \varepsilon^2 $.
  Since interactions decay fast as distance increases, they are truncated at a cutoff radius $r_c$. The potential is then shifted to ensure continuity.
- ### Cell lists (see `forces.py` and `cells.py`)
  Since the interaction is zero at distances $r \ge r_c$, we implement a cell-list approach.
  The domain is partitioned into square cells whose side is greater than the cutoff radius, then interactions are computed only with the neghbouring cells. 
  
## Repository structure
```
.
├── main.py              # Entry point for simulations
├── simulation/          # Core simulation code
│   ├── forces.py        # Wall and particle forces
│   ├── integrator.py    # Velocity-Verlet integrator
│   └── ...             
├── runs/                # Output folder 
├── analysis/            # Post processing utilities
│   ├── analysis.py      # Energy, RDF analysis and plots
│   ├── animation.py     # Create gif animation
└── CONFIG.json          # Simulation parameters
```

## Requirements

This project is written in **Python**.

You will likely need:
- Python 3.x
- Common scientific Python packages (typically `numpy`, `matplotlib`)
  
## Quick start

1. Clone the repository
   ```bash
   git clone https://github.com/lmarini94/Particles
   ``` 
2. Edit the configuration file `CONFIG.json`
3. Run the simulation:
   ```bash
   python main.py
   ```
4. The outputs (`states.npy` and `energies.npy`) are saved in a timestamped folder `run/YYYYMMDD_HHMM`
5. Run the analysis and the animation scripts
   ```bash
   python analysis/analysis.py
   python analysis/animation.py
   ```
6. Graphs and GIF animations are saved in the same timestamped folder `run/YYYYMMDD_HHMM`

## Configuration (`CONFIG.json`)

The configuration file is divided into sections:

- `physical`
  - `N`: number of particles
  - `L`: box size (linear dimension)
  - `d_min`: minimum initialization distance
  - `r_c`: cutoff radius
  - `r_skin`: neighbor-list skin distance
  - `K_0`: initial kietic energy of the system
  - `k_wall`,`delta`: parameter modelling soft wall (stiffness and size of buffer)

- `simulation`
  - `h`: time step
  - `eps`: smoothing parameter for the interparticle interaction
  - `speed`: display/logging stride 
  - `T_MAX`: total simulated time 
  - `seed`: RNG seed (`null` means “not fixed”)

- `model`
  - `pair_potential`: currently set to `"1/r^6 - 1/r^4"`

- `outputs`
  - `states_file`: default `"states.npy"`
  - `energies_file`: default `"energies.npy"`

## Outputs

At the end of a run, the outputs are saved in a timestamped folder `run/YYYYMMDD_HHMM`.
The outputs include:

- `states.npy` which stores a `(T, N, 2)` dimensional array of positions over time
- `energies.npy` which saves all energy components over time (potential due to wall, interactions, kinetic) in a NumPy `.npy`


## Analysis & visualization

The scripts contained in the `analysis/` folder are run by default on the latest simulation, but can be altered to run on a simulation specified by the user.  
