import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import math
from tqdm import tqdm
from numba import jit,njit
from matplotlib import pyplot as plt
import pickle
from modulefile import pickle_object, unpickle_object, map_index
import matplotlib

pandas_full_frame=False
if (pandas_full_frame==True):
    pd.set_option("display.max_rows", None, "display.max_columns", None)


# The idea behind the code is from the PRB article.
# - Compute the N-H bond vector. <br>
# - Project the N-H bond vector directly onto the x-z plane. <br>
# - Also, try projecting the N-H bond vector onto the H-H-H plane and then computing the orientational structure . <br>
# - Plot em. <br>

def get_NH_vectors(sorted_xyz, box=[18.284, 12.740, 11.778], nit=None):
    box = np.array(box)
    NH_vectors_Sim = [] # shape is Nframes, 32, 3 = reshape to Nframes32 * 3
    for timestep in tqdm(sorted_xyz):
        timestep_data = sorted_xyz[timestep].copy(deep=True)
        timestep_data.columns = timestep_data.columns.str.lower()
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


def project_on_XZPlane(NH_vectors):
    """
    Get the projection of a given vector onto the Z axis i.e. along the unit vector 0i + 0j + k.
    The scalar projection on Z is simply the dot product vector. unit_vector_along_z = z_component.
    So this will just get the z_co-ordinate from the vectors.
    """
    NH_vectors_Z = NH_vectors[:, 2]
    NH_vectors_X = NH_vectors[:, 0]
    return NH_vectors_X, NH_vectors_Z


if __name__ == '__main__':
    AA_90k = unpickle_object("../../Traj/R2SCAN_90K/Final-R2SCAN/R2SCAN-90K-Sorted.pkl")
    NH_90k = get_NH_vectors(AA_90k)
    NHXZ = project_on_XZPlane(NH_90k)
    pickle_object(NHXZ, "R2SCAN_OS90K_XZ_Full.pkl")
