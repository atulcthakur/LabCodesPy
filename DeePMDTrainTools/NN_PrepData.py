import numpy as np
import pandas as pd
import scipy as sp
import numba
from tqdm.notebook import tqdm
from numba import jit, njit
from modulefile import xyz_writer, frames_list, extract_times, remove_duplicates
import re, math, os


# Idea:
#
# - Put 90K and 30K pos, force trajectories in the folders.
# - Have to select 3000 frames which are equally spaced in time.
# - Read in the trajectories. Select the 3000 equally spaced frames/ select every 15th frame.
# - Write out the xyz trajectory.
# - Tar and Send.


def master_correct_traj(file, natoms=256):
    listdf = frames_list(file, natoms)
    timelist = extract_times(file)
    corrected_traj = remove_duplicates(listdf, timelist)
    return corrected_traj

def save_n_frames(corrected_traj, nconfs, nevery=None):
    traj = {}
    assert nconfs is not None, "Please input the total number of configs to extract"
    if nevery is None: nevery = math.floor(len(corrected_traj)/nconfs)
    count, num = 0, 0
    for index, key in enumerate(corrected_traj):
        if (index == num):
            traj[key] = corrected_traj[key]
            num += nevery
        if (len(traj) == nconfs): break
    return traj


def PrepForNN(xyz, natoms, nconfs=None, nevery=None, xyzout_name="Pos90.xyz"):
    corrected_xyz = master_correct_traj(xyz, natoms=natoms)
    NN_traj = save_n_frames(corrected_xyz, nconfs=nconfs, nevery=nevery)
    xyz_writer(NN_traj, xyzname=xyzout_name)
    return NN_traj

#Force_90k = PrepForNN("90K/Force_90K.xyz", xyzout_name="NN_Data/90K/Force-90K-NN.xyz")

#Pos_90k = PrepForNN("90K/Pos_90K.xyz", xyzout_name="NN_Data/90K/Pos-90K-NN.xyz")

#------------------------------------------------------------ ENERGIES ---------------------------------------------------------------!!!!

def extract_energies(filename="AA-90K-MD-pos-1.xyz"):
    energylist = []
    file = open(filename, "r")
    for line in tqdm(file):
        if re.search("time", line):
            energy = float(line.split()[-1])
            energylist.append(energy)
    return energylist

def master_correct_energy(file):
    energylist = extract_energies(file)
    timelist = extract_times(file)
    corrected_energy_list = remove_duplicates(energylist, timelist)
    return corrected_energy_list

def write_energies(outfile, corrected_energies):
    with open(outfile, 'a') as file:
        file.write("Time [fs], Energy [Hartree] \n")
        for time, energy in corrected_energies.items():
            time = int(time.split("_")[-1])
            #print(time, energy)
            file.write(f"{time: 10d}, {energy: 17.10f}\n")
    return None

def PrepEnergiesForNN(xyz, nconfs=None, nevery=None, fileout="Energies_NN.dat"):
    corrected_energies = master_correct_energy(xyz)
    NN_traj = save_n_frames(corrected_energies, nconfs=nconfs, nevery=nevery)
    write_energies(fileout, NN_traj)
    return NN_traj

#------------------------------------------------TRINING SETS---------------------------------------------
if __name__ == '__main__':
    # 200K


if __name__ == '__main__':
    # 600K

#------------------------------------------------VALIDATION SETS------------------------------------------
