import numpy as np
import pandas as pd
import scipy as sp
import numba
from tqdm import tqdm
from numba import jit, njit
from modulefile import prepare_xyz_with_virtual_site, xyz_writer, make_whole, sort_traj, add_virtual_site, map_index, read_aimd_pimd_xyz_faster, pickle_object, unpickle_object
from numba.typed import List
from scipy import signal
import scipy as sp
import pickle
import os
import matplotlib.pyplot as plt

k_b = 0.008314

print_full_dataframe=False
if print_full_dataframe==True:
    pd.set_option("display.max_rows", None, "display.max_columns", None)

def fractional_trajectory(traj, xbox, ybox, zbox):
    """
    Converts Processed Trajectory in Cartesian Co-Ordinates to Fractional Co-Ordinates.
    """
    fract = {}
    for time in traj:
        df = traj[time]
        df['x'] = df['x']/xbox
        df['y'] = df['y']/ybox
        df['z'] = df['z']/zbox
        fract[time] = df
    return fract

def prepare_xyz(xyzname=None, box=[18.284, 12.740, 11.778]):
    """
    Takes in the AIMD trajectory, sorts it and returns the sorted trajectory.
    """
    xyz_framed, natoms = read_aimd_pimd_xyz_faster(xyzname) # Reads in the normal AIMD traj and natoms.
    oneframe = {"time_0": xyz_framed["time_0"]} # First frame of the xyz trajectory.
    sorted_oneframe = sort_traj(oneframe, atoms=natoms, x=box[0], y=box[1], z=box[2], cutoff=1.5) # Sorts out the first frame in the trajectory.
    sorted_xyz = map_index(xyz_framed, sorted_df=sorted_oneframe["time_0"]) # Sorts out the reamining frames.
    return sorted_xyz

def get_NH_vectors(sorted_xyz, box=[18.284, 12.740, 11.778], nit=None):
    box = np.array(box)
    NH_vectors_Sim = [] # shape is Nframes, 32, 3 = reshape to Nframes32 * 3
    for timestep in tqdm(sorted_xyz):
        timestep_data = sorted_xyz[timestep].copy(deep=True)
        N_Frame = timestep_data.loc[( timestep_data['atoms'] == "N" )]
        #N_Frame = N_Frame.iloc[[nit]] # Uncomment this is nit is not none.
        NH3_mols = timestep_data.loc[( timestep_data['mols'].isin(N_Frame['mols']) )]
        #print(NH3_mols)
        H3_frame = NH3_mols.loc[(NH3_mols['atoms'] == "H")]
        N_atoms, H_atoms = N_Frame[['x', 'y', 'z']].to_numpy(), H3_frame[['x', 'y', 'z']].to_numpy()
        N_atoms = np.repeat(N_atoms, 3, axis=0)
        NH_vectors = compute_vectors(N_atoms, H_atoms, box) # Shape is 32*3
        NH_vectors_Sim.append(NH_vectors)
        #print(NH_vectors.shape)
    return np.array(NH_vectors_Sim).reshape(len(sorted_xyz)*len(NH_vectors), 3)

@njit
def compute_vectors(N_atoms, H_atoms, box):
    NH_vectors = np.zeros_like(N_atoms)
    for index, (hydrogen, nitrogen) in enumerate(zip(H_atoms, N_atoms)):
        NH = hydrogen - nitrogen
        NH_Min_Img = NH - (box * np.round( (NH/box), 0, np.zeros_like(NH)))
        norm = np.linalg.norm(NH_Min_Img)
        NH_normed = NH_Min_Img/norm
        NH_vectors[index] = NH_normed
    return NH_vectors

#@njit
def project_Z(NH_vectors):
    """
    Get the projection of a given vector onto the Z axis i.e. along the unit vector 0i + 0j + k.
    The scalar projection on Z is simply the dot product vector. unit_vector_along_z = z_component.
    So this will just get the z_co-ordinate from the vectors.
    """
    NH_vectors_Z = NH_vectors[:, 2]
    return NH_vectors_Z

def get_FES(Z_components):
    hist, edges = np.histogram(Z_components, bins=500, density=False)
    return hist, edges

def master_FES(xyz="Pos_90K.xyz", box=[18.284, 12.740, 11.778], Temp=None):
    sorted_xyz = prepare_xyz(xyz, box)
    NHs = get_NH_vectors(sorted_xyz, box)
    Zs = project_Z(NHs)
    Hist, edges = get_FES(Zs)
    edges = (edges[1:] + edges[: -1])/2
    cos_theta = np.arccos(edges)*180/np.pi
    FE =  -np.log((Hist))
    if Temp is not None:
        kb = 0.008314
        FE = kb * Temp * FE
    return cos_theta, FE

## The following function are all meant for analysis of the FE Landscape.
def Get_minima(sorted_xyz, box=[18.284, 12.740, 11.778], mins=[18, 50, 76, 103, 126, 161]):
    """
    This will extract the configurations of the system ammonia belonging to each minima.
    """
    box = np.array(box)
    min1_configs, min2_configs, min3_configs = {}, {}, {}
    for time in tqdm(sorted_xyz):
        timestep_data = sorted_xyz[time].copy(deep=True)
        N_Frame = timestep_data.loc[( timestep_data['atoms'] == "N" )]
        CC_Frame = timestep_data.loc[( timestep_data['atoms'] == "C" )]
        C2H2_frame = timestep_data.loc[( timestep_data['mols'].isin(CC_Frame['mols']))]  # C2H2 frame.
        #for ammonia in range(total_ammonia):
        #N_Frame_for_ith_ammonia = N_Frame.iloc[[ammonia]]
        NH3_mols = timestep_data.loc[( timestep_data['mols'].isin(N_Frame['mols']))] # NH3 molecule
        H3_frame = NH3_mols.loc[(NH3_mols['atoms'] == "H")]
        N_Frame = N_Frame.loc[N_Frame.index.repeat(3)]
        N_atoms, H_atoms = N_Frame[['x', 'y', 'z']].to_numpy(), H3_frame[['x', 'y', 'z']].to_numpy()
        #N_atoms = np.repeat(N_atoms, 3, axis=0)
        NH_vectors = compute_vectors(N_atoms, H_atoms, box) # Shape is 3
        thetas = np.arccos(project_Z(NH_vectors)) * 180/np.pi # 1d array. Dimension is same as H, N
        assert len(thetas) == len(N_atoms) == len(H3_frame) == len(NH_vectors), "Consistency Check"
        min1 = [ index for index, thet in enumerate(thetas) if mins[0]<thet<mins[1] ]
        min2 = [ index for index, thet in enumerate(thetas) if mins[2]<thet<mins[3] ]
        min3 = [ index for index, thet in enumerate(thetas) if mins[4]<thet<mins[5] ]
        #print( pd.concat( [C2H2_frame, N_Frame.iloc[[0]] ] ) )
        #now get those entries from the dataframe and concat it with the cc_Dataframe. so you've nh and c2h2.
        min1_configs[time] = [ pd.concat([C2H2_frame, N_Frame.iloc[[index]], H3_frame.iloc[[index]]]) for index in min1 ]
        min2_configs[time] = [pd.concat([C2H2_frame, N_Frame.iloc[[index]], H3_frame.iloc[[index]]]) for index in min2]
        min3_configs[time] = [pd.concat([C2H2_frame, N_Frame.iloc[[index]], H3_frame.iloc[[index]]]) for index in min3]
        #break
            #get the nh vector, projection, angle, add the structure to the minimum.
    return min1_configs, min2_configs, min3_configs

def Get_minima_with_NH3(sorted_xyz, minimum, box=[18.284, 12.740, 11.778]):
    """
    This will extract the configurations of the system ammonia belonging to each minima.
    """
    box = np.array(box)
    min_configs =  {}
    counter = 0
    assert len(sorted_xyz) == len(minimum), "Both should have same lengths"
    for time in tqdm(sorted_xyz):
        timestep_data = sorted_xyz[time].copy(deep=True)
        min_data = minimum[time]
        min_nh3_data = []
        for structure in min_data:
            #print(structure)
            N_data = structure.loc[( structure['atoms'] == "N" )]
            NH3_mols = timestep_data.loc[( timestep_data['mols'].isin(N_data['mols']))] # NH3 molecule
            NH3_mols = NH3_mols.sort_values(by="atoms")
            frame = pd.concat([structure[:-2] , NH3_mols])
            min_nh3_data.append(frame)
            #print(frame)
        #print(min_nh3_data)
        min_configs[time] = min_nh3_data
        break
        #counter += 1
        #if counter == 20: break
    return min_configs

def H_bonding_at_minima(minimum, accptr="X", donors="N", box=[18.284, 12.740, 11.778]):
    """
    This will just extract the H-bonding at minimum.
    """
    box = np.array(box)
    HB_Data = {}
    for time in tqdm(minimum):
        configs = minimum[time]
        HB_Pattern = []
        for structure in configs:
            Acp_sites = (structure.loc[structure["atoms"] == accptr])[["x", "y", "z"]].to_numpy()
            frame_dnr = structure.loc[structure["atoms"] == donors]
            Dnr_sites, Dnr_mols = frame_dnr[["x", "y", "z"]].to_numpy(), frame_dnr['mols'].to_numpy().astype(np.int32)
            Dnrs_full_mols = structure.loc[ structure['mols'].isin(frame_dnr['mols']) ]
            H_frame = Dnrs_full_mols.loc[(Dnrs_full_mols['atoms'] == "H")]
            H_sites, H_mols = H_frame[["x", "y", "z"]].to_numpy(), H_frame['mols'].to_numpy().astype(np.int32)
            HBs = distance_angle_fun_modified(Acp_sites, Dnr_sites, H_sites, Dnr_mols, H_mols, box, 4, 150)
            HB_Pattern.append(list(HBs))
        HB_Data[time] = HB_Pattern
    return HB_Data

@jit(nopython=True)
def distance_angle_fun_modified(acceptors, donors, hydrogens, donor_mol_array, hydrogen_mol_array, box, distance_cutoff, angle_cutoff):
    assert len(donors) == len(donor_mol_array),"Check your donor arrays"
    #assert len(acceptors), "Check your acceptor arrays"
    assert len(hydrogens) == len(hydrogen_mol_array), "Check your hydrogen arrays"
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
                            #out_list.append(( dnr_ix[i], acptr_ix[j], h_ix[donor_hydrogens_index[k]], 1)) # Append donor, acceptor, and H position along with 1 because of H-bonding. This positions can be used as it to verify H-bond in VMD.
                            out_list.append(1)
                        #else:
                            #continue
                        #    out_list.append([ dnr_ix[i], acptr_ix[j], h_ix[donor_hydrogens_index[k]], 0])
                         #   out_list.append(0)
    if len(out_list) == 0: out_list.append(0)
    return out_list # List containing all triplets involved in H-bonding at a perticular timestep.

def make_histogram(HB_Data):
    """
    This will make the histogram of the HB data.
    """
    Hist_Data = list(np.concatenate(list(HB_Data.values())).flat)
    Histogram, edges = np.histogram(Hist_Data, bins=[-1, 0, 1, 2])
    likeliness = sum(Hist_Data) / len(Hist_Data)

    return Histogram,edges, likeliness

def custom_xyz_writer(minimum_configs, file="min1.xyz"):
    """
    Custom XYZ writer for writing structures created here
    """
    for counter, time in enumerate(minimum_configs):
        configs = minimum_configs[time]
        #print(configs)
        mod_xyz_writer(configs, atoms=None, xyzname=file)
        break
        #if counter == 10: break
    return None

def mod_xyz_writer(input_traj, atoms=None, xyzname="output.xyz", com_traj=False):
    """
    Assumes that atom order and total number of atoms remains constant throughout the trajectory - A resonable assumption but can certainly fail in certain conditions.
    """
    if atoms is None:
        #print(input_traj)
        listdfs = input_traj
        atoms = len(listdfs[0])
    if com_traj:
        atom_array = np.array(["A"]*atoms)
    else:
        atom_array = listdfs[0]['atoms'].to_numpy()
    str_array = np.zeros(atom_array.size, dtype=[('var1', 'U6'), ('var2', np.float64), ('var3', np.float64), ('var4', np.float64)])
    with open(xyzname, 'a') as file:
        for counter, key in enumerate((input_traj)):
            df = key
            #atom_array = df["atoms"].to_numpy() # Lifting the constant atom name order assumption by replacing the original order with the current one.
            file.write("{}\n".format(atoms))
            file.write("time = {}\n".format(counter))
            str_array['var1'] = atom_array
            str_array['var2'] = df['x'].to_numpy()
            str_array['var3'] = df['y'].to_numpy()
            str_array['var4'] = df['z'].to_numpy()
            np.savetxt(file, str_array, fmt='%5s %17.10f %17.10f %17.10f')
            #if counter==10:
            #    break
    return None

def compute_NH3_distribution(NH3_minima_traj, box):
    """
    This will compute the other two NH vectors for a specific minima.
    """
    box = np.array(box)
    NH_angles = []
    for time in tqdm(NH3_minima_traj):
        frame = NH3_minima_traj[time].copy()
        for conf in frame:
            NH3_mol = conf[-4:] # Last four entries are NH3 molecule.
            N_atoms, H_atoms = NH3_mol.loc[NH3_mol["atoms"] == "N"][["x", "y", "z"]].to_numpy(), NH3_mol.loc[NH3_mol["atoms"] == "H"][["x", "y", "z"]].to_numpy()
            N_atoms = np.repeat(N_atoms, 3, axis=0)
            NHs = compute_vectors(N_atoms, H_atoms, box)
            Thetas = np.arccos(project_Z(NHs)) * 180/np.pi
            Angles = np.sort(Thetas)[[0, -1]] # np.array( [angle for angle in Thetas if ((76 > angle) or (angle > 103))] )
            NH_angles.append(Angles)
    #hist, edges = np.histogram(np.array(NH_angles).flatten(), bins=500, range=(0, 180))
    return np.array(NH_angles)


# Free Energy Landscape for the AA-ES Paper:
if __name__ == '__main__':
    print("Prepping the XYZ...")
    xyz_90k = prepare_xyz_with_virtual_site(xyzname="../../Traj/90K/Final-90K/AIMD_90k_AAcocrystal.xyz", box=[18.284, 12.740, 11.778])
    print("Computing Bond Vectors...")
    NHs_90k = get_NH_vectors(xyz_90k, box=[18.284, 12.740, 11.778])
    print("Creating Histrograms..")
    Zs_90k = project_Z(NHs_90k)
    Hist_90k, edges_90k = get_FES(Zs_90k)
    edges_90k = (edges_90k[1:] + edges_90k[: -1])/2
    pickle_object((edges_90k, Hist_90k), "fes-90k.pkl")

if __name__ == '__main__':
    print("Using Fractional XYZ...")
    xyz_90k_fract = fractional_trajectory(xyz_90k, 18.284, 12.740, 11.778)
    print("Computing Bond Vectors...")
    NHs_90k_fract = get_NH_vectors(xyz_90k_fract, box=[1, 1, 1])
    print("Creating Histrograms..")
    Zs_90k_fract = project_Z(NHs_90k_fract)
    Hist_90k_fract, edges_90k_fract = get_FES(Zs_90k_fract)
    edges_90k_fract = (edges_90k_fract[1:] + edges_90k_fract[: -1])/2
    pickle_object((edges_90k_fract, Hist_90k_fract), "fes-90k_fract.pkl")
