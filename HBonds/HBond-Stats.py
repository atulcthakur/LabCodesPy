import numpy as np
import pandas as pd
import scipy as sp
import numba
from tqdm import tqdm
from numba import jit, njit
from modulefile import xyz_writer, make_whole, sort_traj, add_virtual_site, map_index, read_aimd_pimd_xyz_faster, pickle_object, unpickle_object
from numba.typed import List
from scipy import signal
import scipy as sp
import pickle
import matplotlib.pyplot as plt

print_full_dataframe=True
if print_full_dataframe==True:
    pd.set_option("display.max_rows", None, "display.max_columns", None)


# Normalised
# Take legth of the coorelation as a function argument.
def scipy_correlate(array):
    assert type(array) is np.ndarray,"Your code broke because of wrong inputs"
    c_t = sp.signal.correlate(array,array,mode='full',method='auto') #method= 'auto'/'fft'/'direct'
    half = c_t.size//2 # or len(c_t)//2
    c_t = c_t[half:]
    cor_time = len(array)-1 # time normalization factor array length
    #print(cor_time)
    #print(np.linspace(len(array), len(array)-cor_time, cor_time+1, dtype=np.float_))
    c_t = c_t / np.linspace(len(array), len(array)-cor_time, cor_time+1, dtype=np.float_) # Time normalization factor
    #c_t = c_t / np.linspace(nstep,nstep-nt,nt+1,dtype=np.float_) # nnstep = np.linspace(len(array), len(array)-cor_time, cor_time+1, dtype=np.float_)
    return c_t/c_t[0]

mass_dict={'N': 14.0067, 'H': 1.00784, 'C': 12.0107}

def prepare_xyz(xyzname="Pos/Pos_90k.xyz", box=[18.284, 12.740, 11.778], sorted_pkl_file=None):
    """
    Takes in the AIMD trajectory, sorts it, fixes the broken molecules and then adds the virtual site at their center of mass.
    Returns the trajectory with the virtual site.
    Only works for AIMD so far though extension for PIMD should be straightforward.
    """
    if sorted_pkl_file is None:
        xyz_framed, natoms = read_aimd_pimd_xyz_faster(xyzname)# Reads in the normal AIMD traj and natoms.
        oneframe = {"time_0": xyz_framed["time_0"]} # First frame of the xyz trajectory.
        sorted_oneframe = sort_traj(oneframe, atoms=natoms, x=box[0], y=box[1], z=box[2], cutoff=1.5) # Sorts out the first frame in the trajectory.
        sorted_xyz = map_index(xyz_framed, sorted_df=sorted_oneframe["time_0"]) # Sorts out the reamining frames.
    else:
        sorted_xyz = unpickle_object(sorted_pkl_file)
        natoms = len(sorted_xyz['time_0'])
    wholed_xyz = make_whole(sorted_xyz, atoms=natoms, x=box[0], y=box[1], z=box[2]) # This fixes the molecules broken acorss periodic boundaries for the full trajectory. Slower. Maybe write a numba alternative ?
    virtual_xyz = add_virtual_site(wholed_xyz, molecules=["C","H"], masses=mass_dict) # Adds in the virtual site on the acetyelene molecules.
    return virtual_xyz # Returns the virtual site added trajectory.


def H_bonds(structure, donors="C", acceptors="N", hydrogens="H", distance_cutoff=None, angle_cutoff=None, x=18.284, y=12.740, z=11.778):
    """
    Takes in the structure trajectory with/without virtual site. Evaluates the H-bonds between donors and acceptors based on the specified distance and angle criterion.
    Returns dictionary listing all the H-bonds found at each timestep.
    Configured only for AIMD for right now.
    """
    # global acptr_pos, dnr_pos, H_pos, dnr_mols, H_mols, box
    HBond_Stat = {} # Output dictionary. Contains the list of H-bonds found at a given timestep.
    box = np.array([x, y, z]) # Box array.
    for counter, key in enumerate(tqdm(structure)):  # Iterating over all times.
        data = structure[key].copy(deep=True) # Current Dataframe.
        donor_frame = data.loc[(data['atoms'] == donors)].reset_index() # Making the donor Dataframe, i.e C for instance.
        acceptor_frame = data.loc[(data['atoms'] == acceptors)].reset_index() # Making the acceptor Dataframe, i.e. N for instance.
        molecule_df = data.loc[(data['mols'].isin(donor_frame['mols']))]  # Molecule dataframe for donor molecules.
        hydrogen_frame = molecule_df.loc[(molecule_df['atoms'] == hydrogens)].reset_index()  # Hydrogens of the donor molecules.
        # Selecting the donor hydrogens from the molecule dataframe
        # After this we're just creating inputs for function distance_angle(...)
        acptr_pos = acceptor_frame[['x', 'y', 'z']].to_numpy() # Creating numpy array of acceptor positions
        dnr_pos = donor_frame[['x', 'y', 'z']].to_numpy() # Creating numpy array of donor positions
        H_pos = hydrogen_frame[['x', 'y', 'z']].to_numpy() # Creating numpy array of donor hydrogen positions
        dnr_mols = donor_frame['mols'].to_numpy().astype(np.int32) # Creating numpy array of donor molecules id.
        H_mols = hydrogen_frame['mols'].to_numpy().astype(np.int32) # Creating numpy array of hydrogen moleculde id
        # For testing purpose only.
        acptr_ix = acceptor_frame['level_0'].to_numpy() # The position indicator of the acceptor atom in the sorted xyz file.
        dnr_ix = donor_frame['level_0'].to_numpy() # == | ==
        h_ix = hydrogen_frame['level_0'].to_numpy() # == | ==
        HBs = distance_angle_fun(acptr_pos, dnr_pos, H_pos, dnr_mols, H_mols, box, distance_cutoff, angle_cutoff, acptr_ix, dnr_ix, h_ix) # finding out the H-bonded pairs.
        #HBs = (1, 2)
        HBond_Stat[key] = list(HBs) # Transforming numba list to python list and storing in the H_bond_stat dictionary.
        #if (counter==0): break # for testing
    return HBond_Stat


@jit(nopython=True)
def distance_angle_fun(acceptors, donors, hydrogens, donor_mol_array, hydrogen_mol_array, box, distance_cutoff, angle_cutoff, acptr_ix, dnr_ix, h_ix):
    assert len(donors) == len(donor_mol_array) == len(dnr_ix),"Check your donor arrays"
    assert len(acceptors) == len(acptr_ix), "Check your acceptor arrays"
    assert len(hydrogens) == len(hydrogen_mol_array) == len(h_ix), "Check your hydrogen arrays"
    out_list = List() # Empty output list object (numba typed)
    for i in range(len(donors)): # Donor atoms are C, for instance.
        for j in range(len(acceptors)): #  Accptor atoms are N, for instance.
            d_ij = acceptors[j] - donors[i] # Creating the distance vector, so this would be C-N / N-X vector in this case.
            r_ij = d_ij - (box * np.round((d_ij)/box, 0, np.zeros_like(d_ij)))  # Creating the minimum image distance vector. The np.zeros_like is just because numba requires to call it that way.
            acptr_dnr_distance = np.linalg.norm(r_ij)  # Minimum image distance
            if acptr_dnr_distance <= distance_cutoff: # Distance criterion between acceptor and donor.
                mol_number = donor_mol_array[i] # Molecule id to which the current donor belongs.
                #donor_hydrogens = hydrogens[hydrogen_mol_array==mol_number] # Selecting the hydrogens of the donor atom.
                donor_hydrogens_index = (hydrogen_mol_array==mol_number).nonzero()[0] # Mask on the hydrogens of the donor atom
                donor_hydrogens = hydrogens[donor_hydrogens_index] # Applying the mask to select H which belong to the same molecule as donor.
                for k in range(len(donor_hydrogens)): # Iterating over donor hydrogens
                    d_ik =  donor_hydrogens[k] - donors[i] # Distance vector between hydrogen and donor, so C-H / N-H vector for this case.
                    r_ik = d_ik - (box * np.round((d_ik)/box, 0, np.zeros_like(d_ik))) # It's minimum image vector.
                    norm = np.linalg.norm(r_ik)  # Norm of the above vector.
                    if norm <= 1.2: # Distance criterion between donor atom and it's Hydrogen atom.
                        r_ha = donor_hydrogens[k] -  acceptors[j]
                        r_ha = r_ha - (box * np.round((r_ha)/box, 0, np.zeros_like(d_ij)))
                        n_rha = np.linalg.norm(r_ha)
                        dot = np.dot( (r_ik/norm), (r_ha/n_rha))  # dot product of the two unit vectors.
                        angle = np.arccos(dot) * (180/np.pi)  # Angle in degrees between r_ik and r_ij i.e angle between C-H vector and C-N vector for instance.
                        #print(norm)
                        if angle >= angle_cutoff: # If angle is within the tolerance
                            #continue
                            #print(angle)
                            out_list.append(( dnr_ix[i], acptr_ix[j], h_ix[donor_hydrogens_index[k]], 1)) # Append donor, acceptor, and H position along with 1 because of H-bonding. This positions can be used as it to verify H-bond in VMD.
                            #out_list.append([i, j, k, 1])
                        #else:
                            #continue
                        #    out_list.append([ dnr_ix[i], acptr_ix[j], h_ix[donor_hydrogens_index[k]], 0])
                            #out_list.append([i, j, k, 0])
    return out_list # List containing all triplets involved in H-bonding at a perticular timestep.


def find_unique_pairs(HBond_Stat):
    """
    Accepts output of the Hbond function which is a dictionary containing list of triplets involved in H-bonding at a perticular timestep.
    Returns a set of all the unique pairs.
    """
    unique_triplets = set()
    for key in tqdm(HBond_Stat):
        triplets = HBond_Stat[key]
        for trips in triplets:
            unique_triplets.add(trips)
    return unique_triplets


def create_timeseries(unique_triplets, HBond_Stat):
    """
    Accepts output of the find_unique_pairs function which is a set containing all the unique triplets.
    The second input is the HBond_Stat dictionary used as the input of find_unique_pairs function.
    Returns timeseries of all the unique triplets.
    """
    n_triplets = len(unique_triplets)
    nsteps = len(HBond_Stat)
    alltrips_timeseries = {}
    for trips in tqdm(unique_triplets):
        trips_timeseries = np.zeros(nsteps)
        for index, time in enumerate(HBond_Stat):
            time_triplets = HBond_Stat[time]
            if trips in time_triplets:
                trips_timeseries[index] = 1
        alltrips_timeseries[trips] = trips_timeseries
    assert len(alltrips_timeseries) == n_triplets, "Something went wrong. All triplets are not present in the output."
    return alltrips_timeseries


def merge_timeseries(trips_ts1, trips_ts2):
    all_ts = trips_ts1 | trips_ts2
    assert len(all_ts) == len(trips_ts1) + len(trips_ts2), "Some elements got replaced. Check !"
    return all_ts


def correlate_Hbonds(alltrips_timeseries):
    c_t = 0
    for trips_ts in tqdm(alltrips_timeseries):
        timeseries = alltrips_timeseries[trips_ts]
        c_t += scipy_correlate(timeseries)
    c_t_norm = c_t / len(alltrips_timeseries)
    return c_t_norm


def get_Rate(c_t):
    """
    According to Luzar and Chandler, k(t) is = -dc(t)/dt.
    The c(t) is actually a time correlation defined on fluctuations of 'h' operator.
    However, it can be analytically shown that the the time derivative of c(t) is same as the one you'd get from time derivative of TCF of operator 'h'.
    """
    c_t_dot = np.diff(c_t)
    k_t = -c_t_dot
    return k_t

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
        np.savetxt(np.c_[ct_final.to_numpy()], fmt="%15.10f")
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
        np.savetxt(np.c_[ct_final.to_numpy()], fmt="%15.10f")
    return ct_final

def Master(AIMD_traj, **kwargs):
    HBonds = H_bonds(AIMD_traj, **kwargs)
    unique_HBs = find_unique_pairs(HBonds)
    HB_timeseries = create_timeseries(unique_HBs, HBonds)
    #HB_C_t = correlate_Hbonds(HB_timeseries)
    HB_intermittent = Intermittent_Loop_Implementation(HB_timeseries, save=False, Name=None)
    HB_continous = Cont_Loop_Implementation(HB_timeseries, save=False, Name=None)
    #time = np.arange(len(HB_C_t))
    return HB_intermittent, HB_continous

# R2scan 90k K H-Bond Dynamics For C-H...N and N-H...X H-bonds. To be run on amarel.
if __name__ == '__main__':
    print("Preparing...")
    AIMD_90k_wannier = prepare_xyz(xyzname=None, box=[18.284, 12.740, 11.778], sorted_pkl_file="R2SCAN-90K-Sorted.pkl")
    print("CHN H-Bonds...")
    CHN_HB = Master(AIMD_90k_wannier, donors="C", acceptors="N", hydrogens="H", distance_cutoff=4, angle_cutoff=150, x=18.284, y=12.740, z=11.778)
    pickle_object(CHN_HB, "CHN_HB_Ct.pkl")
    print("NHX H-Bonds...")
    NHX_HB = Master(AIMD_90k_wannier, donors="N", acceptors="X", hydrogens="H", distance_cutoff=4, angle_cutoff=150, x=18.284, y=12.740, z=11.778)
    pickle_object(NHX_HB, "NHX_HB_Ct.pkl")
