import numpy as np
import pandas as pd
import pickle
import math
from tqdm import tqdm
from numba import jit, njit, prange
from matplotlib import pyplot as plt
from modulefile import read_aimd_pimd_xyz_faster, make_whole, sort_traj, map_index, xyz_writer, chunk_traj, read_sorted, pickle_object, unpickle_object
from scipy.signal import savgol_filter


pandas_full_frame=False
if (pandas_full_frame==True):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

# This is the Final Trial For the Angular Velocity Correlation.
# The idea is to compute the angular velcity correlation here and then compare it to the one you get from LAMMPS as well as for the model system which I've to set up.

# The Pseudocode goes as:
#
# - Drop a vector from N and perpendicular to the H-H-H plane. This will be pointing down so reverse its direction by taking negative of it which should point up. <br>
# - Now draw a vector towards one H and originating at N. Now the above two vectors are not perpendicular so you make'em perpendicular by using GS. <br>
# - You can then construct the third perpendicular vector using gram-schmidt process again. <br>
# - Normalise the vectors and you've the system of orthonormal axes which originate at the N and one of them points upward in the direction of principle axis of NH3.  <br>
# - Construct the trajectory of such vectors for all ammonia molecules in your system at each timestep. <br>
# - Take its numerical derivative and compute angular velcoties using the wiki formula for rigid bodies. <br>
# - Now take the simple P1 correlation of it and you're set and done. <br>
# - Enjoy ! <br>

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
    return c_t #/c_t[0] I changed this here to return a non-normalized correlation (it is still normalized by the lag time factor).

@njit
def construct_equation_of_plane(H_coordinates, box):
    """ This function takes in the coordinates of Hydrogens triplets H, H, H for a frame as an array.
        So it is basically an array shaped as len, 9.
        This function unpacks such array into vectors and then construct the equation of plane from them.
        The final equation of plane is returned as a numpy array containing coefficients.
    """
    number_of_nh3_molecules = len(H_coordinates) # Basically the number of NH3 molecules. Used to group the hydrogens of a nh3 together in the next step.
    input_array = H_coordinates #.reshape(number_of_nh3_molecules, 9) # Reshaping the flattened H-coordiantes array such that the three hydorgens of a NH3 now belong to a single row vector of length 9.
    plane = np.zeros((number_of_nh3_molecules, 4)) # Empty array to store the final equation of the plane containing a, b, c and d coefficients.
    for index, molecule in enumerate(input_array): # Iterating over each individual hydrogen triplets belonging to a ammonia molecule.
        H1, H2, H3 = np.split(molecule, 3) # Spliting the 9 sized vector into individual cooridnates of H1, H2 and H3.
        H1_H2_vec = H2 - H1 # H1-->H2 minimum image vector.
        H1_H3_vec = H3 - H1 # H1-->H3 minimum image vector.
        cross_prod = np.cross(H1_H2_vec, H1_H3_vec) # Taking the cross product to find the eqaution of a normal to the plane.
        a,b,c = cross_prod  # Storing the components of the normal.
        d = np.sum(cross_prod * H1) # Finding d.
        plane[index] = a, b, c, d # Storing in the array plane
    return plane # Returning the result.

@njit
def find_perpendicular_from_N(N_coords, plane):
    """
    This function will take in the coordinates of N and corresponding equation of planes. It will drop a perpendicular from N to the plane and return its foot say point F.
    This will also find out the FN vector, extend it at point I and then find a NI vector.
    The return value will be the co-ordinates of the normalized NI vector.
    """
    NI_vectors_ = np.zeros_like(N_coords) # creating an exmpty array to store results. N_coords has a shape of (Number of nitrogens, 3)
    for index, (nitrogen, plane) in enumerate(zip(N_coords, plane)):
        # refer here for logic = https://www.geeksforgeeks.org/find-the-foot-of-perpendicular-of-a-point-in-a-3-d-plane/
        x, y, z = nitrogen # unpacking the coordinates.
        #a, b, c, d = plane
        numerator = np.sum(-plane * np.array([x, y, z, 1])) # Numerator on the page linked above.
        denominator = np.sum(plane[:-1] * plane[:-1])  # Denominator given on the page linked above.
        k = numerator / denominator # The constant K which is needed to find the foot point.
        foot = plane[:-1]*k + nitrogen # coordinates of the foot of the perpendicular from nitrogen.
        FN_vector = nitrogen - foot # F-->N minimum image vector.
        FI_vector = 2*FN_vector
        NI_vector = FI_vector - nitrogen
        #NI_vector_normalized = NI_vector / np.linalg.norm(NI_vector)
        NI_vectors_[index] = NI_vector #NI_vector_normalized - no need for normalization
    return NI_vectors_

@jit(nopython=True)
def gram_schmidt(X):
    """
    Implements Gram-Schmidt orthogonalization.

    Parameters
    ----------
    X : an n x k array with linearly independent rows vectors

    Returns
    -------
    U : an n x k array with orthonormal row vectors

    """
    # Converting to column based for implementation
    X = X.T

    # Set up
    n, k = X.shape
    U = np.empty((n, k))
    I = np.eye(n)

    # The first col of U is just the normalized first col of X
    v1 = X[:,0]
    U[:, 0] = v1 / np.sqrt(np.sum(v1 * v1))

    for i in range(1, k):
        # Set up
        b = X[:, i]       # The vector we're going to project
        Z = X[:, 0:i]     # First i-1 columns of X

        # Project onto the orthogonal complement of the col span of Z
        M = I - Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        u = M @ b

        #M = I - np.dot(np.dot(Z, np.linalg.inv(np.dot(Z.T, Z))) , Z.T)
        #u = np.dot(M, b)

        # Normalize
        U[:, i] = u / np.sqrt(np.sum(u * u))

    return U.T

@njit
def create_axes_system(N_coords, H1_coords, H2_coords, NI_vectors):
    """
    This function will return a orthonormal axes system given coordinates of nitrogen, hydrogens and NI vectors.
    """
    orthonormal_axes = np.zeros((len(N_coords), 3, 3)) # Empty array for storing orthonormal vectors.

    # Take in the H frame and create a vector. Now use GS on it.
    # Simply take the cross product to find the third vector.
    # Store it like the other derivative function needs and copy that in here. Compute angular velocities. Do P1 correlation using groups.
    # Verify ! Verify ! Verify !.

    NH1_vector = H1_coords - N_coords # Make sure the H belong the same molecule the N belongs to. This creates a N-H1 vector.
    NH2_vector = H2_coords - N_coords # This creates a N-H2 vector.
    basis_vectors = np.hstack((NI_vectors, NH1_vector, NH2_vector))
    basis_vectors = basis_vectors.reshape(len(basis_vectors), 3, 3) # reshaping it here
    for index, basis in enumerate(basis_vectors):
        orthonormal =  gram_schmidt(basis)
        orthonormal_axes[index] = orthonormal
    assert np.all(orthonormal_axes), "Some elements are still zero. Check your output" # Very Strong test.
    return orthonormal_axes

#@njit
def compute_omega(axes_traj):
    number_of_nh3 = len(axes_traj[0, :, 0, 0])
    steps = len(axes_traj[:, 0, 0, 0])
    all_omegas = np.zeros((number_of_nh3, steps-1, 3))
    for nh3 in range(number_of_nh3):
        nh3_timeseries = axes_traj[:, nh3, :, :]
        time_derivative = np.diff(nh3_timeseries, axis=0)
        e1, e2, e3 = nh3_timeseries[:, 0, :][:-1], nh3_timeseries[:, 1, :][:-1], nh3_timeseries[:, 2, :][:-1]
        e1_dot, e2_dot, e3_dot = time_derivative[:, 0, :], time_derivative[:, 1, :], time_derivative[:, 2, :]
        omega1 =  ((e1_dot * e2).sum(axis=1, keepdims=True)) * e3
        omega2 =  ((e2_dot * e3).sum(axis=1, keepdims=True)) * e1
        omega3 =  ((e3_dot * e1).sum(axis=1, keepdims=True)) * e2
        omega = omega1 + omega2 + omega3
        all_omegas[nh3, :, :] = omega
    assert np.all(all_omegas), "Some elements are still zero. Check your output" # Very Strong test.
    return all_omegas

def P1_correlation(all_omegas):
    x_corr_avg, y_corr_avg, z_corr_avg = 0, 0, 0 # correct
    number_of_nh3 = len(all_omegas) # correct
    for nh3 in range(number_of_nh3): # correct
        #print(all_omegas[nh3, :, 0].shape)
        x_corr_avg += numpy_correlate(all_omegas[nh3, :, 0])
        y_corr_avg += numpy_correlate(all_omegas[nh3, :, 1])
        z_corr_avg += numpy_correlate(all_omegas[nh3, :, 2])

    corr_omega_x = x_corr_avg / (3*number_of_nh3)
    corr_omega_y = y_corr_avg / (3*number_of_nh3)
    corr_omega_z = z_corr_avg / (3*number_of_nh3)
    total_corr = corr_omega_x + corr_omega_y + corr_omega_z
    return total_corr/total_corr[0], corr_omega_x/corr_omega_x[0], corr_omega_y/corr_omega_y[0], corr_omega_z/corr_omega_z[0]


def prepare_from_xyz(xyzfile, box, to_pickle=False):
    """
    Takes xyz, sorts, makes whole, returns wholed/maybe an pickled object.
    """
    trajectory, natoms = read_aimd_pimd_xyz_faster(xyzfile) # full initialized trajectory.
    oneframe = {"time_0": trajectory["time_0"]} # oneframe.
    sorted_oneframe = sort_traj(oneframe, atoms=natoms, x=box[0], y=box[1], z=box[2], cutoff=1.5) # Sort One Frame trajectory.
    sorted_frame = sorted_oneframe["time_0"] # Store the sorted dataframe as sorted_frame.
    sorted_xyz = map_index(trajectory, sorted_df=sorted_frame) # Map the sorted index on the rest of the trajectory.
    wholed_xyz = make_whole(sorted_xyz, atoms=natoms, x=box[0], y=box[1], z=box[2], write_traj=False)
    if to_pickle:
        pickle_object(obj, file)
        return "Successfully Pickled"
    else:
        return wholed_xyz, natoms


def do_angular_velocity(pklfile, box):
    """
    Handle all the calls here.
    Call prepare, and the other functions here. You may have to handle the stuff in between the calls here too.
    """
    box = np.array(box)
    #wholed_traj, natoms = prepare_from_xyz(xyzfile, box, to_pickle=False)
    #wholed_traj = unpickle_object(pklfile)
    wholed_traj = pklfile
    for index, time in enumerate(tqdm(wholed_traj)):
        data = wholed_traj[time].copy(deep=True)
        # Select N, H1 and H2 frames. Make sure the atoms from the molecules are aligned in rows.
        N_df = data.loc[ data['atoms']=="N" ]  # Selecting nitrogen atoms.
        N_array = N_df[["x", "y", "z"]].to_numpy()
        NH3_df = data.loc[data['mols'].isin(N_df['mols'])] # NH3-dataframe
        Nmask = (NH3_df['atoms']=="N")
        H_dataframe = NH3_df.loc[~(Nmask)]
        H1_array = H_dataframe.groupby("mols").nth(0)[["x", "y", "z"]].to_numpy()
        H2_array = H_dataframe.groupby("mols").nth(1)[["x", "y", "z"]].to_numpy()
        H3_array = H_dataframe.groupby("mols").nth(2)[["x", "y", "z"]].to_numpy()
        all_H_array = np.hstack((H1_array, H2_array, H3_array))
        eqn_plane = construct_equation_of_plane(all_H_array, box)
        N_foot = find_perpendicular_from_N(N_array, eqn_plane)
        orthonormal_system = create_axes_system(N_array, H1_array, H2_array, N_foot)
        if (index==0): orthonormal_axes_trajetory = np.zeros((len(wholed_traj), len(N_array), 3, 3))
        orthonormal_axes_trajetory[index] = orthonormal_system
        #break
    omegas = compute_omega(orthonormal_axes_trajetory)
    omega_P1 = P1_correlation(omegas)
    return omega_P1

def set_column_names(traj):
    frame = traj["time_0"]
    cols =  list(frame.columns)
    frame_cols_lower = [col.lower() for col in cols]
    traj_out = {}
    for time in traj:
        df = traj[time]
        df.columns = frame_cols_lower
        traj_out[time] = df
    return traj


phz_to_cminv = 100/2.9979245800
fhz_to_cminv = 100000/2.9979245800
def fft(array, conv=None, **kwargs):
    """
    I'll take FFT of any array you give me.
    """

    fft_array = np.abs(np.fft.fft(array))**2
    fft_freq = np.fft.fftfreq(array.shape[-1], **kwargs)
    if conv is not None:
        fft_freq = conv*fft_freq
    return np.array_split(fft_freq, 2)[0], np.array_split(fft_array, 2)[0]


#Analysis for E&S

if __name__ == '1st':
    AA_AIMD_90K_unpickled = unpickle_object("../../Traj/90K/Final-90K/90K_full_sorted.pkl")
    AA_90k = set_column_names(AA_AIMD_90K_unpickled)
    AA_wholed = make_whole(AA_90k, atoms=256, x=18.284, y=12.740, z=11.778, write_traj=False)
    pickle_object(AA_wholed, "../../Traj/90K/wholed_90k.pkl")

if __name__ == '2nd':
    AA_wholed = unpickle_object("../../Traj/90K/Final-90K/wholed_90k.pkl")
    P1_omega_all= do_angular_velocity(AA_wholed, [18.284, 12.740, 11.778])
    pickle_object(P1_omega_all, "P1_omega_all.pkl")

if __name__ == '__main__':
    tot_omega, Xw, Yw, Zw = unpickle_object("P1_omega_all.pkl")
    # Tot
    power = fft(tot_omega, conv=fhz_to_cminv)
    pickle_object(power, "P1_total_omega_PowerSpectrum.pkl")
    # X
    power_X = fft(Xw, conv=fhz_to_cminv)
    pickle_object(power_X, "X_omega_PowerSpectrum.pkl")
    # Y
    power_Y = fft(Yw, conv=fhz_to_cminv)
    pickle_object(power_Y, "Y_omega_PowerSpectrum.pkl")
    # Z
    power_Z = fft(Zw, conv=fhz_to_cminv)
    pickle_object(power_Z, "Z_omega_PowerSpectrum.pkl")
