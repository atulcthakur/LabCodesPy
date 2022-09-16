# All the import statements go here. Modulefile is a custom module.
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
    return c_t #/c_t[0]

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
        molecules = np.repeat(np.arange(0, len(ethane_carbons)/factor, dtype=np.int16), factor) # Remove 2 for methane
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

def OLD_P1Thetaintermittent(theta_ts):
    corr = 0
    for mol_id in theta_ts:
        mol_ts = theta_ts[mol_id]
        corr += scipy_correlate(mol_ts)
    corr_normalized = corr / len(theta_ts)
    return corr_normalized

def OLD_Theta_for_continuous(theta_ts):
    chi_ts = {}
    for mol_id in tqdm(theta_ts):
        mol_ts = theta_ts[mol_id]
        #full_len_mol_ts = len(mol_ts)
        cont_mol_ts_avg = 0
        for length in range(len(mol_ts)):
            mol_ts_new_origin = mol_ts[length:]
            cont_mol_ts = []
            for index in range(1, len(mol_ts_new_origin)+1):
                chi = np.prod(mol_ts_new_origin[:index])
                cont_mol_ts.append(chi)
            #print(len(cont_mol_ts))
            #print(len(np.pad(np.array(cont_mol_ts), (0, length))))
            cont_mol_ts_avg += np.pad(np.array(cont_mol_ts), (0, length))
        chi_ts[mol_id] = cont_mol_ts_avg/ len(mol_ts)
        #tot = np.sum(chi_ts.values(), axis=1)
    return chi_ts

def OLD_Continuous_Correlation_Theta(theta_original_timeseries):
    """

    """
    cont_corr, correlation_sum = 0, 0
    for molecule in tqdm(theta_original_timeseries):
        mol_ts = theta_original_timeseries[molecule]
        for time_origin in range(len(mol_ts)):
            theta_array = mol_ts[time_origin:]
            correlation_at_one_time_origin = np.cumprod(theta_array)
            #print(theta_array)
            #print(correlation_at_one_time_origin)
            #break
            correlation_sum += np.pad(correlation_at_one_time_origin, (0, time_origin))
        #break
        time_origin_averaged_correlation_for_one_molecule = correlation_sum / np.linspace(len(correlation_sum), 1, len(correlation_sum))
        cont_corr += time_origin_averaged_correlation_for_one_molecule
    cont_corr_averaged_over_molecules = cont_corr / len(theta_original_timeseries)
    cont_corr_normalized = cont_corr_averaged_over_molecules / cont_corr_averaged_over_molecules[0]
    return cont_corr_normalized

def check(theta_original_timeseries):
    ct_over_all_molecules = 0
    for molecule in tqdm(theta_original_timeseries):
        mol_ts = theta_original_timeseries[molecule]
        full_ct_over_all_time_origins = 0
        for index, time_origin in enumerate(range(len(mol_ts))):
            h_ts = mol_ts[time_origin:]
            full_ct = []
            for lag_time in range(len(h_ts)):
                ct_lag = h_ts[0]*h_ts[lag_time]
                full_ct.append(ct_lag)
            full_ct_over_all_time_origins += np.pad(full_ct, (0, index))
            #break
        avg_full_ct_over_all_time_origins = full_ct_over_all_time_origins/ np.linspace(len(full_ct_over_all_time_origins), 1, len(full_ct_over_all_time_origins))
        ct_over_all_molecules += avg_full_ct_over_all_time_origins
    avg_ct = ct_over_all_molecules/ct_over_all_molecules[0]
    return avg_ct, ct_over_all_molecules


def Intermittent_Loop_Implementation(theta_original_timeseries, save=False, Name=None):
    ct_over_all_molecules_and_time_origins, count = 0, 0
    nframes = len(theta_original_timeseries[list(theta_original_timeseries.keys())[0]]) # Finding out total number of frames in the trajectory.
    for time_origin in tqdm(range(0, nframes)):
        time_origin_theta = {key: value[count:] for key, value in theta_original_timeseries.items()}
        #print(time_origin_theta[2])
        number_of_molecules_with_time, full_ct_over_all_molecules_for_one_time_origin  = 0, 0
        for molecule in time_origin_theta:
            h_ts = time_origin_theta[molecule]
            number_of_molecules_with_time = number_of_molecules_with_time + h_ts
            full_ct = []
            for lag_time in range(len(h_ts)):
                ct_lag = h_ts[0]*h_ts[lag_time]
                full_ct.append(ct_lag)
            full_ct_over_all_molecules_for_one_time_origin = full_ct_over_all_molecules_for_one_time_origin + np.array(full_ct)
        ct_one_time_origin_averaged_over_molecules =  full_ct_over_all_molecules_for_one_time_origin/ number_of_molecules_with_time
        ct_over_all_molecules_and_time_origins = ct_over_all_molecules_and_time_origins + np.pad(ct_one_time_origin_averaged_over_molecules, (0, count))
        count = count + 1
    ct_final = ct_over_all_molecules_and_time_origins/ np.linspace(len(ct_over_all_molecules_and_time_origins), 1, len(ct_over_all_molecules_and_time_origins))
    if save is True and Name is not None:
        np.savetxt(Name, np.c_[ct_final], fmt="%15.10f")
    return ct_final

def Cont_Loop_Implementation(theta_original_timeseries, save=False, Name=None):
    ct_over_all_molecules_and_time_origins, count = 0, 0
    nframes = len(theta_original_timeseries[list(theta_original_timeseries.keys())[0]]) # Finding out total number of frames in the trajectory.
    for time_origin in tqdm(range(0, nframes)):
        time_origin_theta = {key: value[count:] for key, value in theta_original_timeseries.items()}
        number_of_molecules_with_time, full_ct_over_all_molecules_for_one_time_origin  = 0, 0
        for molecule in time_origin_theta:
            h_ts = time_origin_theta[molecule]
            number_of_molecules_with_time = number_of_molecules_with_time + h_ts
            full_ct = []
            H_TS = np.cumprod(h_ts)
            for lag_time in range(len(h_ts)):
                ct_lag = h_ts[0]*H_TS[lag_time]
                full_ct.append(ct_lag)
            full_ct_over_all_molecules_for_one_time_origin = full_ct_over_all_molecules_for_one_time_origin + np.array(full_ct)
        ct_one_time_origin_averaged_over_molecules =  full_ct_over_all_molecules_for_one_time_origin / number_of_molecules_with_time
        ct_over_all_molecules_and_time_origins = ct_over_all_molecules_and_time_origins + np.pad(ct_one_time_origin_averaged_over_molecules, (0, count))
        count = count + 1
        #if (count == 20): break
    ct_final = ct_over_all_molecules_and_time_origins/ np.linspace(len(ct_over_all_molecules_and_time_origins), 1, len(ct_over_all_molecules_and_time_origins))
    if save is True and Name is not None:
        np.savetxt(Name, np.c_[ct_final], fmt="%15.10f")
    return ct_final


def P1_Theta_intermittent(theta_ts, save=False, Name=None):
    corr = 0
    for mol_id in theta_ts:
        mol_ts = theta_ts[mol_id]
        corr += scipy_correlate(mol_ts)
    #norm = np.sum(list(theta_ts.values()), axis=0)
    corr_norm = corr/corr[0]
    if save is True and Name is not None:
        np.savetxt(Name, np.c_[corr_norm], fmt="%15.10f")
    return corr_norm

def get_avg_h(theta_original_timeseries):
    avg_h_moleculewise = []
    for mol in theta_original_timeseries:
        molecule_prob = theta_original_timeseries[mol].mean()
        avg_h_moleculewise.append(molecule_prob)
    total_avg_h = sum(avg_h_moleculewise) / len(avg_h_moleculewise)
    return total_avg_h

def delta_intermittent_Loop_Implementation(theta_original_timeseries, avg_h, save=False, Name=None):
    ct_over_all_molecules_and_time_origins, count = 0, 0
    total_molecules = len(theta_original_timeseries)
    nframes = len(theta_original_timeseries[list(theta_original_timeseries.keys())[0]]) # Finding out total number of frames in the trajectory.
    for time_origin in tqdm(range(0, nframes)):
        time_origin_theta = {key: value[count:] for key, value in theta_original_timeseries.items()}
        #print(time_origin_theta[2])
        number_of_molecules_with_time, full_ct_over_all_molecules_for_one_time_origin  = 0, 0
        for molecule in time_origin_theta:
            h_ts = time_origin_theta[molecule]
            h_ts = h_ts - avg_h
            #number_of_molecules_with_time = number_of_molecules_with_time + h_ts
            full_ct = []
            for lag_time in range(len(h_ts)):
                ct_lag = h_ts[0] * h_ts[lag_time]
                full_ct.append(ct_lag)
            full_ct_over_all_molecules_for_one_time_origin = full_ct_over_all_molecules_for_one_time_origin + np.array(full_ct)
            #print(full_ct_over_all_molecules_for_one_time_origin)
        ct_one_time_origin_averaged_over_molecules =  full_ct_over_all_molecules_for_one_time_origin / total_molecules # / number_of_molecules_with_time
        ct_over_all_molecules_and_time_origins = ct_over_all_molecules_and_time_origins + np.pad(ct_one_time_origin_averaged_over_molecules, (0, count))
        count = count + 1
    ct_final = ct_over_all_molecules_and_time_origins / np.linspace(len(ct_over_all_molecules_and_time_origins), 1, len(ct_over_all_molecules_and_time_origins))
    if save is True and Name is not None:
        np.savetxt(Name, np.c_[ct_final], fmt="%15.10f")
    return ct_final

def delta_P1_Theta_intermittent(theta_ts, avg_h, save=False, Name=None): # FASTER NON LOOP METHOD TO DO DELTA CORRELATION.
    corr = 0
    for mol_id in theta_ts:
        mol_ts = theta_ts[mol_id]
        mol_ts -= avg_h
        corr += scipy_correlate(mol_ts)
    corr_norm = corr / len(theta_ts)
    if save is True and Name is not None:
        np.savetxt(Name, np.c_[corr_norm], fmt="%15.10f")
    return corr_norm

def master(xyzname, molecule="ethane", z_hi=5.7, z_lo=0):
    mol, natoms = read_aimd_pimd_xyz_faster(xyzname)
    if molecule == "methane": print("Please remember to change things for methane.")
    mol_carbons = get_carbons(mol, molecule=molecule)
    Carbons_from_zero = substract_box(mol_carbons, box_lo=[-16.4952, -17.6852, -24.1489])
    mol_firstlayer, listmols = select_first_layer(Carbons_from_zero, molecule=molecule, z_hi=z_hi, z_lo=z_lo)
    theta = make_theta(mol_firstlayer, listmols)
    # theta_corr = P1_Theta_intermittent(theta, save=False, Name=None)  # Ct using Atul's method (Fast). Change save to True and put in the name for the file.
    # ct_new_inter = Intermittent_Loop_Implementation(theta, save=False, Name=None) # Ct using loops (Slow). Change save and name.
    # ct_new_cont = Cont_Loop_Implementation(theta, save=False, Name=None) # Ct continuous using loops (Slow). Change save and name.
    avg_h = get_avg_h(theta)
    delta_ct_test = delta_P1_Theta_intermittent(theta, avg_h, save=True, Name="TestDelta.dat")
    return None



if __name__ == '__main__':
    Delta = master(xyzname="../mixture-3-7-nvt-prod2-94K.xyz", molecule="ethane", z_hi=5.7, z_lo=0)
