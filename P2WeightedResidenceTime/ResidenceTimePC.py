import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import numba
from tqdm.notebook import tqdm
from numba import jit, njit
from useful import read_aimd_pimd_xyz_faster, xyz_writer, make_whole, sort_traj, map_index
from numba.typed import List
from numba.types import float64, int64
import pickle
import os, shutil
import matplotlib.pyplot as plt

# Normalised
def scipy_correlate(array):
    assert type(array) is np.ndarray,"Your code broke because of wrong inputs"
    c_t = sp.signal.correlate(array,array,mode='full',method='auto') #method= 'auto'/'fft'/'direct'
    half = c_t.size//2
    c_t = c_t[half:]
    cor_time = len(array)-1
    c_t = c_t / np.linspace(len(array), len(array)-cor_time, cor_time+1, dtype=np.float_) # Time normalization factor
    return c_t/c_t[0]


def get_carbons(traj, molecule=None):
    out = {}
    if molecule == "ethane":
        factor = 2
    elif molecule == "methane":
        factor = 1
    else:
        return "Put either methane or ethane for molecule"
    for counter, time in enumerate(tqdm(traj)):
        frame = traj[time].copy(deep=True)
        CC1 = frame[1080: (1080+4096)].copy(deep=True) # Change for methane
        CC2 = frame[6256: (6256+4096)].copy(deep=True)
        CC3 = frame[11432: (11432+4096)].copy(deep=True)
        ethane_frame = pd.concat([CC1, CC2, CC3], ignore_index=True)
        ethane_carbons = ethane_frame.loc[ethane_frame['atoms'] == 'C'].reset_index(drop=True)
        molecules = np.repeat(np.arange(0, len(ethane_carbons)/factor, dtype=np.int16), 2) # Remove 2 for methane
        ethane_carbons['mols'] = molecules
        out[time] = ethane_carbons
    return out

def substract_box(traj, box_lo=[-16.4952, -17.6852, -24.1489]):
    out = {}
    box_lo = np.array(box_lo)
    print("Please make sure the box_lo dimensions are", box_lo)
    for counter, time in enumerate(tqdm(traj)):
        frame = traj[time].copy(deep=True)
        frame[['x', 'y', 'z']] =  frame[['x', 'y', 'z']] - box_lo
        out[time] = frame
    return out

def select_first_layer(traj, z_hi=5.7, z_lo=0, molecule=None):
    out = {}
    unique_mols = []
    if molecule == "ethane":
        factor = 2
    elif molecule == "methane":
        factor = 1
    else:
        return "Put either methane or ethane for molecule"
    for counter, time in enumerate(tqdm(traj)):
        frame = traj[time].copy(deep=True)
        frame.columns= frame.columns.str.lower()
        frame_com = (frame.groupby('mols').sum()/factor).reset_index() # Remove /2 for methane mols.
        first_layer = frame_com.loc[ (z_hi >= frame_com['z']) & (frame_com['z'] >= z_lo)]
        out[time] = first_layer[['mols', 'z']].reset_index(drop=True)
        first_layer_mols = first_layer['mols'].to_numpy()
        unique_mols.append(first_layer_mols)
    unique_mols_allframes = np.unique(np.concatenate(unique_mols).flatten())
    return out, unique_mols_allframes


def make_theta(traj, unique_mols):
    nsteps = len(traj)
    out = {mol_id: np.zeros(nsteps) for mol_id in unique_mols}
    for counter, time in enumerate(tqdm(traj)):
        frame = traj[time].copy(deep=True)
        frame_mols = frame['mols'].to_numpy()
        for mol_id in out:
            if mol_id in frame_mols:
                out[mol_id][counter] = 1
    return out

def P1_Theta_intermittent(theta_ts):
    corr = 0
    for mol_id in theta_ts:
        mol_ts = theta_ts[mol_id]
        corr += scipy_correlate(mol_ts)
    corr_normalized = corr / len(theta_ts)
    return corr_normalized


def P1_Theta_continuous(theta_ts):
    corr = 0
    for mol_id in theta_ts:
        mol_ts = theta_ts[mol_id]

        corr += scipy_correlate(mol_ts)
    return None

def master():
    pass
