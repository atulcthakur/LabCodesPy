import numpy as np
import pandas as pd
import math
import scipy as sp
from tqdm import tqdm
from numba import jit
from matplotlib import pyplot as plt
import os
import warnings
from useful import read_aimd_pimd_xyz_faster, pickle_object, unpickle_object, xyz_writer, make_whole, sort_traj, map_index
from ase.io import read, write
import ase
from ase.units import Bohr, Rydberg, kJ, kB, kcal, mol, fs, Angstrom, Hartree
import random


Hartree_to_kcalpermol = Hartree * mol / kcal
Forces_HartreeperBohr_to_kcalpermolangstrom = Hartree_to_kcalpermol/ Bohr

def forces_units(force_traj, units=1.0):
    forces_units = {}
    for time in tqdm(force_traj):
        force_df = force_traj[time].copy(deep=True)
        force_df['x'] =  force_df['x'] * units
        force_df['y'] =  force_df['y'] * units
        force_df['z'] =  force_df['z'] * units
        forces_units[time] = force_df
    return forces_units


def read_energies(file, units=None):
    energy = pd.read_csv(file, comment="T", header=None)
    energy_list = list(energy.to_numpy()[:, 1])
    if units is not None:
        energy_list = list(energy.to_numpy()[:, 1] * units)
    return energy_list


def extxyz_pos_force_writer(input_traj, force_traj, energies, atoms=None, xyzname="output.xyz", com_traj=False, box= [1.0, 1.0, 1.0]):
    """
    # I'm assuming that you kept the ordering of atoms same in both position and force traj.
    """
    if atoms is None:
        listdfs = list(input_traj.values())
        atoms = len(listdfs[0])
    if com_traj:
        atom_array = np.array(["A"]*atoms)
    else:
        atom_array = listdfs[0]['atoms'].to_numpy()
    str_array = np.zeros(atom_array.size, dtype=[('var1', 'U6'), ('var2', np.float64), ('var3', np.float64), ('var4', np.float64),     ('var5', np.float64), ('var6', np.float64), ('var7', np.float64)])
    with open(xyzname, 'a') as file:
        for counter, key in enumerate(tqdm(input_traj)):
            df = input_traj[key]
            df_force = force_traj[key]
            energy = energies[counter]
            file.write("{}\n".format(atoms))
            comment = f"Lattice=\"{box[0]} 0.0 0.0 0.0 {box[1]} 0.0 0.0 0.0 {box[2]}\" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} pbc=\"T T T\" Time={float(counter)}\n"
            file.write(comment) # "time = {}\n".format(counter))
            str_array['var1'] = atom_array
            str_array['var2'] = df['x'].to_numpy()
            str_array['var3'] = df['y'].to_numpy()
            str_array['var4'] = df['z'].to_numpy()
            str_array['var5'] = df_force['x'].to_numpy()
            str_array['var6'] = df_force['y'].to_numpy()
            str_array['var7'] = df_force['z'].to_numpy()
            np.savetxt(file, str_array, fmt='%5s %17.10f %17.10f %17.10f %17.10f %17.10f %17.10f')
            #if counter==2: break
    return None


def merge_multiple_trajs(**args):
    """
    Pass me multiple dictinary trajectories and I'll merge them in order without losing any frames.
    """
    out = {}
    item_list = []
    for item in args:
        item_list = list(item.values())
    for index, frame in enumerate(item_list):
        key = 'time_' + str(index)
        out[key] = frame
    return out

#------------------------------------------------------------------------------------------------------------------------------------------------------

#I used concat.sh to concat everything.
#Then next step are:

if __name__ == '__main__':
    pos, _ = read_aimd_pimd_xyz_faster("Pos-Master.xyz")
    forces, _ = read_aimd_pimd_xyz_faster("Force-Master.xyz")
    En = read_energies("En-Master.dat", units=Hartree_to_kcalpermol)
    print(len(En), len(pos))
    forces_kcalpermolang = forces_units(forces, units=Forces_HartreeperBohr_to_kcalpermolangstrom)
    extxyz_pos_force_writer(pos, forces_kcalpermolang, En, box=[18.284, 12.740, 11.778], xyzname="AA_Temp.extxyz")
