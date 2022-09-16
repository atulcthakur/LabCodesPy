import numpy as np
import pandas as pd
from scipy import signal
import scipy as sp
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from numba import jit
import sys
import matplotlib.pyplot as plt
import pickle
from useful import pickle_object, unpickle_object


# So the code will go like this:
# 0) Read up the velocity according to your format.
# 1) Sort the velocities according to the dataframe.
# 2) Calculate the COM velocity for a given molecule.
# 3) Export the data for correlation P1 and P2.


pd.set_option("display.max_rows", None, "display.max_columns", None)


@jit(nopython=True)
def direct_correlate(array1,array2,array3):
    assert len(array1)==len(array2)==len(array3),"Your code broke because of wrong inputs"
    #print("This code will calculate P2 correlation only")
    t_cor = len(array1)
    t_run = len(array1)
    c_t = np.zeros(t_cor)
    for time in range(t_cor):
        t_max =  t_run - time
        #c_t[time] = sum(array[0:t_max]*array[time:t_max+time]) / t_max
        c_t[time] = np.sum(((array1[0:t_max]*array1[time:t_max+time]) + (array2[0:t_max]*array2[time:t_max+time]) + (array3[0:t_max]*array3[time:t_max+time]))**2) / t_max
    return (3/2)*c_t-(1/2)

def numpy_correlate(array, cor_time=None):
    time = len(array)
    if cor_time is None:
        cor_time=time-1
    assert (cor_time < time), "Make sure cor_time is less than length of your input array minus one"
    assert type(array) is np.ndarray,"Your code broke because of wrong inputs"
    c_t = np.correlate(array,array,mode='full')
    half = c_t.size//2
    c_t = c_t[half:half+cor_time+1] # half+max_time_for_corr_function+1] # what is nt ?
    c_t = c_t / np.linspace(time, time-cor_time, cor_time+1, dtype=np.float_) # Time normalization factor
    #c_t = c_t / np.linspace(nstep,nstep-nt,nt+1,dtype=np.float_) # nstep = np.linspace(len(array), len(array)-cor_time, cor_time+1, dtype=np.float_)
    return c_t/c_t[0] # Returning normalized version

def numpy_correlate_unnormalized(array, cor_time=None):
    time = len(array)
    if cor_time is None:
        cor_time=time-1
    assert (cor_time < time), "Make sure cor_time is less than length of your input array minus one"
    assert type(array) is np.ndarray,"Your code broke because of wrong inputs"
    c_t = np.correlate(array,array,mode='full')
    half = c_t.size//2
    c_t = c_t[half:half+cor_time+1] # half+max_time_for_corr_function+1] # what is nt ?
    c_t = c_t / np.linspace(time, time-cor_time, cor_time+1, dtype=np.float_) # Time normalization factor
    #c_t = c_t / np.linspace(nstep,nstep-nt,nt+1,dtype=np.float_) # nstep = np.linspace(len(array), len(array)-cor_time, cor_time+1, dtype=np.float_)
    return c_t # Returning un-normalized version

class read:
    """
    This contains the methods to read the velocity and the sorted trajectory.
    Make sure that the position trajectory is duplicate corrected and sorted. Also, the velocity trajectory is required to be duplicate corrected.
    """
    def __init__(self, pos_sorted, vel_xyz, atoms):
        self.pos_sorted = pos_sorted
        self.vel_xyz = vel_xyz
        self.atoms = atoms

    def read_duplicate_corrected_velocity_traj(self):
        print("For now, this method is only capable of reading duplicate corrected velocity file. The extension should be straightforward")
        datatypes = {"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64}
        xyz_chunk = pd.read_csv(self.vel_xyz, dtype=datatypes, chunksize=self.atoms, usecols=[0,1,2,3], header=0, names=['atoms', 'x', 'y', 'z'])
        self.output = {}
        for counter, chunk in tqdm(enumerate(xyz_chunk)):
            chunk.reset_index(inplace=True)
            chunk['index'] = np.arange(self.atoms)
            chunk.insert(1,'mols', np.zeros(self.atoms, dtype=int))
            self.output[('').join(['time_',str(counter)])] = chunk
            #if counter == 10:
            #    break
        return self.output

    def chunk_traj(self):
        datatypes = {"index": np.int32, "mols" : np.int32 ,"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64}
        self.df_chunk = pd.read_csv(self.pos_sorted, usecols=[0,1,2,3,4,5], header=0, names=['index', 'mols', 'atoms', 'x', 'y', 'z'], chunksize=self.atoms, dtype=datatypes)
        return self

    def read_sorted(self, File_parser=None):
        if File_parser is None:
            File_parser=self.df_chunk
        self.traj_dict = {}
        for counter, chunk in tqdm(enumerate(File_parser)):
            chunk.reset_index(drop=True, inplace=True)
            self.traj_dict[('').join(['time_',str(counter)])] = chunk
        return self.traj_dict

class vacf:
    """
    This will calculate the velocity autocorrelation function for molecules.
    """
    def __init__(self, sorted_trajectory, vel_trajectory, mass={"C": 12, "H": 1, "N": 14}, molecules= ["N","H"], length_mol=4, total_atoms=256):
        self.mass = mass
        self.molecules = molecules
        self.length_mol = length_mol
        self.total_atoms = total_atoms
        self.trajectory = sorted_trajectory
        self.vel_trajectory = vel_trajectory
        assert self.molecules[0] != "H" ,"First atom should be heavy and not H"

    def sort_velocity(self, sorted_index_df=None, write_output=False, filename="sorted_velocities.csv"):
        if sorted_index_df is None:
            sorted_index_df=self.trajectory['time_0']
        self.sorted_vel_traj = {} # Creating the blank dictionary to append the sorted results
        sorted_index = sorted_index_df['index'] # index using which we want to sort every dataframe in the trajectory
        for key in tqdm(self.vel_trajectory):
            curr_vdf = self.vel_trajectory[key]
            sort_vdf = curr_vdf.reindex(sorted_index)
            sort_vdf.reset_index(drop=True, inplace=True) # now
            sort_vdf['mols'] = sorted_index_df['mols']
            self.sorted_vel_traj[key] = sort_vdf # Append the modified dataframe to the new dictionary
        if write_output:
            list_vdfs = list(self.sorted_vel_traj.values())
            (pd.concat(list_vdfs)).to_csv(filename, index=False)
        return self

    def velocity_com(self):
        com_traj = []
        self.com_timeseries = {}
        for counter, key in enumerate(tqdm(self.sorted_vel_traj)):
            df = self.sorted_vel_traj[key].copy(deep=True)
            heavy_atom_df = df.loc[(df['atoms'] == self.molecules[0])] # Dataframe of heavy atoms corresponding to 1st position in molecule list
            repeat_count = df.pivot_table(index=['mols'], aggfunc='size') # How many times each entry in mols columns is repeated i.e. effectively the length of the molecule
            repeat_count = (repeat_count.mask(repeat_count != self.length_mol)).dropna() # If the length of the molecule is not as specified in the functions argument then drop them.
            mol_df = df.loc[(df['mols'].isin(heavy_atom_df['mols'])) & (df['mols'].isin(repeat_count.index))] # Dataframe containing the molecules having the specified heavy atom and corresponding to the specified length
            mol_df['mass'] = mol_df['atoms'].map(self.mass) # Map the masses to respective atoms as given in specified mass dictionary (input)
            assert (not mol_df['mass'].isnull().any()), "Some of the masses have not been assigned which means your mass dictionary is incomplete. Please provide mass list of all unique atoms in the trajectory."
            mol_df[['x','y','z']] = mol_df[['x','y','z']].mul(mol_df['mass'], axis=0) #  Multiplying by the masses
            com_df = mol_df.groupby("mols").sum() # Numerator of the COM formula i.e x1*m1 + x2*m2 + x3*m3..... and same along y and z for every molecule.
            com_df[['x','y','z']]  = com_df[['x','y','z']].div(com_df['mass'], axis=0) # Dividing by the total mass to get x_cm, y_cm and z_cm
            com_traj.append(com_df)
            #if counter==10:
            #   break
        all_com_dfs = pd.concat(com_traj)
        grouped = all_com_dfs.groupby('mols')
        for name, group in grouped: # iterating over the grouped object we created.
            self.com_timeseries[name] = group # adding each name and group as key:value pair in the dictionary. Each group is timeseries of an atom.
        assert len(grouped) == len(self.com_timeseries) == len(com_traj[0]), "Something got replaced" # Consistency check. This makes sure that no replacement occured while creating groups or writing timerseries dictionary. Each atom should have its own entry.
        return self # Returning the final dictionary.

    def velocity_all_com(self):
        com_traj = []
        self.com_timeseries = {}
        for counter, key in enumerate(tqdm(self.sorted_vel_traj)):
            df = self.sorted_vel_traj[key].copy(deep=True)
            #heavy_atom_df = df.loc[(df['atoms'] == self.molecules[0])] # Dataframe of heavy atoms corresponding to 1st position in molecule list
            #repeat_count = df.pivot_table(index=['mols'], aggfunc='size') # How many times each entry in mols columns is repeated i.e. effectively the length of the molecule
            #repeat_count = (repeat_count.mask(repeat_count != self.length_mol)).dropna() # If the length of the molecule is not as specified in the functions argument then drop them.
            #mol_df = df.loc[(df['mols'].isin(heavy_atom_df['mols'])) & (df['mols'].isin(repeat_count.index))] # Dataframe containing the molecules having the specified heavy atom and corresponding to the specified length
            mol_df = df
            mol_df['mass'] = mol_df['atoms'].map(self.mass) # Map the masses to respective atoms as given in specified mass dictionary (input)
            assert (not mol_df['mass'].isnull().any()), "Some of the masses have not been assigned which means your mass dictionary is incomplete. Please provide mass list of all unique atoms in the trajectory."
            mol_df[['x','y','z']] = mol_df[['x','y','z']].mul(mol_df['mass'], axis=0) #  Multiplying by the masses
            com_df = mol_df.groupby("mols").sum() # Numerator of the COM formula i.e x1*m1 + x2*m2 + x3*m3..... and same along y and z for every molecule.
            com_df[['x','y','z']]  = com_df[['x','y','z']].div(com_df['mass'], axis=0) # Dividing by the total mass to get x_cm, y_cm and z_cm
            #print(com_df, mol_df)
            com_traj.append(com_df)
            #if counter==0:
            #    break
        all_com_dfs = pd.concat(com_traj)
        grouped = all_com_dfs.groupby('mols')
        for name, group in grouped: # iterating over the grouped object we created.
            self.com_timeseries[name] = group # adding each name and group as key:value pair in the dictionary. Each group is timeseries of an atom.
        assert len(grouped) == len(self.com_timeseries) == len(com_traj[0]), "Something got replaced" # Consistency check. This makes sure that no replacement occured while creating groups or writing timerseries dictionary. Each atom should have its own entry.
        return self # Returning the final dictionary.


    def velocity_atoms(self):
        atom_traj = [] # creating a blank list for appending the heavy_atom_dfs. List because in the end we're going to use pd.concat.
        self.atom_timeseries = {} # Creating a blank dictionary for appending the timeseries dataframe of each atom. This is the return object.
        for counter, key in tqdm(enumerate(self.sorted_vel_traj)): # Looping over sorted velocity trajectory.
            df = self.sorted_vel_traj[key] # current frame under consideration
            heavy_atom_df = df.loc[(df['atoms'] == self.molecules[0])] # Locate heavy atom df i.e entries where "atoms" column is same as 1st entry of molecule list.
            atom_traj.append(heavy_atom_df) # Append the heavy atom df to the blank list created earlier.
            #if counter==5:
            #    break
        all_dfs = pd.concat(atom_traj) # Now stack the dataframes stored in atom_traj in order over one another.
        grouped = all_dfs.groupby('index') # Grouping according index column which should have unique entries for each atom. This'll basically select out a time series dataframe of an atom with unique index i.
        for name, group in grouped: # iterating over the grouped object we created.
            self.atom_timeseries[name] = group # adding each name and group as key:value pair in the dictionary. Each group is timeseries of an atom.
        assert len(grouped) == len(self.atom_timeseries) == len(atom_traj[0]), "Something got replaced" # Consistency check. This makes sure that no replacement occured while creating groups or writing timerseries dictionary. Each atom should have its own entry.
        return self # Returning the final dictionary.

    def velocity_all_atoms(self):
        atom_traj = list(self.sorted_vel_traj.values())
        all_dfs = pd.concat(atom_traj, ignore_index=True)
        grouped = all_dfs.groupby('index')
        self.all_atom_timeseries = {}
        for name, group in grouped:
            self.all_atom_timeseries[name] = group
        assert len(grouped) == len(self.all_atom_timeseries) == len(atom_traj[0]), "Something Got Replaced. Debug!"
        return self

    def do_P2_vacf(self, for_atoms=False, for_COMs=False, for_allatoms=False):
        correlate = 0
        if for_atoms:
            self.time_trajectory = self.atom_timeseries
        elif for_COMs:
            self.time_trajectory = self.com_timeseries
        elif for_allatoms:
            self.time_trajectory = self.all_atom_timeseries
        else:
            raise Exception("Either for_atoms or for_COM or for_allatoms needs to be True")
        for key in tqdm(self.time_trajectory):
            curr_ts = self.time_trajectory[key]
            x_array = curr_ts.loc[:, 'x'].to_numpy()
            y_array = curr_ts.loc[:, 'y'].to_numpy()
            z_array = curr_ts.loc[:, 'z'].to_numpy()
            correlate += direct_correlate(x_array, y_array, z_array)
        self.normed_correlation = correlate / len(self.time_trajectory)
        return self.normed_correlation

    def do_P1_vacf(self, for_atoms=False, for_COMs=False, for_allatoms=False, time=None):
        correlate_x, correlate_y, correlate_z = 0, 0, 0
        # Take the input as a traj and caculate p1_vacf, then average and then return normalized. Should be similar to P2.
        if for_atoms:
            self.time_trajectory = self.atom_timeseries # timeseries of each atom in the trajectory which belongs to self.molecules
        elif for_COMs:
            self.time_trajectory = self.com_timeseries
        elif for_allatoms:
            self.time_trajectory = self.all_atom_timeseries
        else:
            raise Exception("Either for_atoms or for_COM or for_allatoms needs to be True")
        for key in tqdm(self.time_trajectory):
            curr_ts = self.time_trajectory[key]
            x_array = curr_ts.loc[:, 'x'].to_numpy()
            y_array = curr_ts.loc[:, 'y'].to_numpy()
            z_array = curr_ts.loc[:, 'z'].to_numpy()
            correlate_x += numpy_correlate(x_array, cor_time=time)
            correlate_y += numpy_correlate(y_array, cor_time=time)
            correlate_z += numpy_correlate(z_array, cor_time=time)
        avg_corr_x = correlate_x / len(self.time_trajectory)
        avg_corr_y = correlate_y / len(self.time_trajectory)
        avg_corr_z = correlate_z / len(self.time_trajectory)
        self.total_correlation = (avg_corr_x + avg_corr_y + avg_corr_z) / 3.0
        return self.total_correlation

    def do_P1_vacf_unnormalized(self, for_atoms=False, for_COMs=False, for_allatoms=False, time=None):
        correlate_x, correlate_y, correlate_z = 0, 0, 0
        # Take the input as a traj and calculate p1_vacf, then average and then return normalized. Should be similar to P2.
        if for_atoms:
            self.time_trajectory = self.atom_timeseries # timeseries of each atom in the trajectory which belongs to self.molecules
        elif for_COMs:
            self.time_trajectory = self.com_timeseries
        elif for_allatoms:
            self.time_trajectory = self.all_atom_timeseries
        else:
            raise Exception("Either for_atoms or for_COM or for_allatoms needs to be True")
        for key in tqdm(self.time_trajectory):
            curr_ts = self.time_trajectory[key]
            x_array = curr_ts.loc[:, 'x'].to_numpy()
            y_array = curr_ts.loc[:, 'y'].to_numpy()
            z_array = curr_ts.loc[:, 'z'].to_numpy()
            correlate_x += numpy_correlate_unnormalized(x_array, cor_time=time)
            correlate_y += numpy_correlate_unnormalized(y_array, cor_time=time)
            correlate_z += numpy_correlate_unnormalized(z_array, cor_time=time)
        avg_corr_x = correlate_x / len(self.time_trajectory)
        avg_corr_y = correlate_y / len(self.time_trajectory)
        avg_corr_z = correlate_z / len(self.time_trajectory)
        self.total_correlation = (avg_corr_x + avg_corr_y + avg_corr_z) / 3.0
        return self.total_correlation


def mirror(arr, axis=0):
    """Mirror array `arr` at index 0 along `axis`.
    The length of the returned array is 2*arr.shape[axis]-1 ."""
    # Shamelessly stolen from pwtool.signal.mirror
    return np.concatenate((arr[::-1],arr[1:]), axis=axis)


phz_to_cminv = 100/2.9979245800
fhz_to_cminv = 100000/2.9979245800
pd.options.mode.chained_assignment = None
def fft(array, conv=None, **kwargs):
    """
    I'll take FFT of any array you give me.
    """
    if pad_zeros:
        fft_array =  np.abs(np.fft.fft(np.pad(array, (0, len(array)), 'constant')))
    else:
        fft_array = np.abs(np.fft.fft(array))
    fft_freq = np.fft.fftfreq(array.shape[-1], **kwargs)
    if conv is not None:
        fft_freq = conv*fft_freq
    return np.array_split(fft_freq, 2)[0], np.array_split(fft_array, 2)[0]


if __name__ == '__main__':
    read_xyzs_90k = read(pos_sorted="../../Traj/90K/Correcting-90K/90K_full_sorted.xyz.csv", vel_xyz="../../Traj/90K/Correcting-90K/Vel_90K_duplicate_removed.csv", atoms=256)
    velocities_90k = read_xyzs_90k.read_duplicate_corrected_velocity_traj()
    positions_90K = unpickle_object("/Users/act114/Box/Lab-Work/My_Papers/AA-Paper/AA-Earth_Space/Figures/Traj/90K/Final-90K/90K_full_sorted.pkl")
    # For All Atoms - Total VACF -----------------------------------------------------------------------------------------------------------------------------------------
    print("Working on Total VACF... ")
    vel_90k_autocorr = vacf(sorted_trajectory=positions_90K, vel_trajectory=velocities_90k)
    velocities_90k_corr = vel_90k_autocorr.sort_velocity().velocity_all_atoms().do_P1_vacf_unnormalized(for_allatoms=True)
    time = np.arange(len(velocities_90k_corr))
    fft_total = fft(velocities_90k_corr, conv=fhz_to_cminv)
    pickle_object((time, velocities_90k_corr, fft_total), "unnormalized/TotalVACF-90-un.pkl" )

    # # For C2H2 VACF -----------------------------------------------------------------------------------------------------------------------------------------------------
    # print("Working on C2H2 VACF... ")
    # vel_90k_autocorr_for_acetylenes = vacf(sorted_trajectory=positions_90K, vel_trajectory=velocities_90k, molecules=['C', 'H'])
    # velocities_90k_C2h2_corr = vel_90k_autocorr_for_acetylenes.sort_velocity().velocity_com().do_P1_vacf(for_COMs=True)
    # time2 = np.arange(len(velocities_90k_C2h2_corr))
    # fft_C2H2 = fft(velocities_90k_C2h2_corr, conv=fhz_to_cminv)
    # pickle_object((time2, velocities_90k_C2h2_corr, fft_C2H2), "C2H2VACF-90.pkl" )
    #
    # # For NH3 VACF -----------------------------------------------------------------------------------------------------------------------------------------------------
    # print("Working on NH3 VACF... ")
    # vel_90k_autocorr_for_nh3 = vacf(sorted_trajectory=positions_90K, vel_trajectory=velocities_90k, molecules=['N', 'H'])
    # velocities_90k_nh3_corr = vel_90k_autocorr_for_nh3.sort_velocity().velocity_com().do_P1_vacf(for_COMs=True)
    # time3 = np.arange(len(velocities_90k_nh3_corr))
    # fft_NH3 = fft(velocities_90k_nh3_corr, conv=fhz_to_cminv)
    # pickle_object((time3, velocities_90k_nh3_corr, fft_NH3), "NH3VACF-90.pkl" )
    #
    # # For Carbon C VACF -----------------------------------------------------------------------------------------------------------------------------------------------------
    # print("Working on C VACF... ")
    # vel_90k_autocorr_for_carbons = vacf(sorted_trajectory=positions_90K, vel_trajectory=velocities_90k, molecules=['C', 'H'])
    # velocities_90k_Carbon_corr = vel_90k_autocorr_for_carbons.sort_velocity().velocity_atoms().do_P1_vacf(for_atoms=True)
    # time4 = np.arange(len(velocities_90k_Carbon_corr))
    # fft_C = fft(velocities_90k_Carbon_corr, conv=fhz_to_cminv)
    # pickle_object((time4, velocities_90k_Carbon_corr, fft_C), "CarbonVACF-90.pkl" )
    #
    # # For Ammonia N VACF -----------------------------------------------------------------------------------------------------------------------------------------------------
    # print("Working on N VACF... ")
    # vel_90k_autocorr_for_nitrogens = vacf(sorted_trajectory=positions_90K, vel_trajectory=velocities_90k, molecules=['N', 'H'])
    # velocities_90k_N_corr = vel_90k_autocorr_for_nitrogens.sort_velocity().velocity_atoms().do_P1_vacf(for_atoms=True)
    # time5 = np.arange(len(velocities_90k_N_corr))
    # fft_N = fft(velocities_90k_N_corr, conv=fhz_to_cminv)
    # pickle_object((time5, velocities_90k_N_corr, fft_N), "NitrogensVACF-90.pkl" )

    # For C2H2 and NH3 COM VACF -----------------------------------------------------------------------------------------------------------------------------------------------------
    print("Working on C2H2-NH3 VACF... ")
    vel_90k_autocorr_for_c2h2_nh3 = vacf(sorted_trajectory=positions_90K, vel_trajectory=velocities_90k, molecules=['C', 'H'])
    velocities_90k_c2h2_nh3_corr = vel_90k_autocorr_for_c2h2_nh3.sort_velocity().velocity_all_com().do_P1_vacf_unnormalized(for_COMs=True)
    time6 = np.arange(len(velocities_90k_c2h2_nh3_corr))
    fft_C2H2_NH3 = fft(velocities_90k_c2h2_nh3_corr, conv=fhz_to_cminv)
    pickle_object((time6, velocities_90k_c2h2_nh3_corr, fft_C2H2_NH3), "unnormalized/C2H2-NH3-VACF-90-un.pkl" )
