import os 
import sys
# import glob
# import time
import logging
from tqdm import tqdm
from itertools import chain

import numpy as np 
import pandas as pd
from numpy import linalg as LA

import parmed as pmd
import MDAnalysis as mda 
from MDAnalysis.analysis import align

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

class grid(object): 
    def __init__(self, x: int, y: int, z: int, grid_size): 
        self.xyz = np.array([x, y, z], dtype=int)
        self.grid_size = grid_size
        self.center = self.get_center()

    def __eq__(self, o: object) -> bool:
        return np.array_equal(self.xyz, o.xyz)

    def __hash__(self) -> int:
        return hash(tuple(self.xyz))

    def __repr__(self) -> str:
        return f"Grid at {self.xyz}"

    def get_center(self): 
        center = (self.xyz + 0.5) * self.grid_size
        return center

    def get_neighbors(self, cutoff: int):
        rad_range = np.arange(-cutoff, cutoff + 1)
        neigh_mesh = np.meshgrid(rad_range, rad_range, rad_range)
        neigh_grids = np.column_stack([i.ravel() for i in neigh_mesh])
        neigh_grids = neigh_grids + self.xyz
        return [grid(*i, self.grid_size) for i in neigh_grids]
        

class atom_grid(grid): 
    def __init__(self, 
            position: np.ndarray,
            radius: float,  
            grid_size: float):
        super().__init__(*position // grid_size, grid_size)
        self.position = position
        self.grid_size = grid_size
        self.radius = radius

    def get_occupied_grids(self):
        cutoff = self.radius // self.grid_size
        neighbors = self.get_neighbors(cutoff) 
        occupied_grids = [grid for grid in neighbors 
                if LA.norm(grid.center - self.position) < self.radius]
        return occupied_grids


class atoms_grids(list): 
    def __init__(self, atoms, top_file:str, grid_size:float):
        self.atoms = atoms 
        self.grid_size = grid_size
        self.top = pmd.load_file(top_file)
        self.atom_radii = self.assign_atom_radii() 
        self.atom_grids = [atom_grid(pos, radius, self.grid_size) 
                for pos, radius in zip(self.atoms.positions, self.atom_radii)]
        self.occupied_grids = set(chain(*[grid.get_occupied_grids() for grid in self.atom_grids]))

    def __len__(self): 
        return len(self.occupied_grids)

    def assign_atom_radii(self): 
        atom_radii = [self.top.atoms[i].sigma/2 for i in self.atoms.indices]
        return np.array(atom_radii)
    