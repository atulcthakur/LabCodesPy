import numpy as np
import pandas as pd
import numba
#from tqdm import tqdm
from numba import jit, njit


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


def read_aimd_pimd_xyz_faster(filename, natoms=None, PIMD=False):
    "Read AIMD/PIMD xyz Faster"
    print(f"Reading File {filename}") # Printing the filename
    if natoms is None: # If natoms is not set by the user then read the file to get natoms.
        with open(filename, 'r') as f:  # opening the file.
            natoms = int(f.readline().strip()) # reading in natoms.
    complete_traj = {}  # Final traj to store the output.
    datatypes = {"atoms" : "str", "x" : np.float32, "y" : np.float32, "z" : np.float32} # Datatypes that we're gonna cast on the colums read from the file.
    if PIMD: # If the trajectory is from i-pi
        comment_symbol = "#"  # set the comment symbol as #.
    else:
        comment_symbol = "i" # else use the following for normal AIMD cp2k trajectories. (Maybe different for Lammps xyz files.)
    dataframe = pd.read_csv(filename, usecols=[0,1,2,3], delim_whitespace=True, names=['atoms','x','y','z'], dtype=datatypes, comment="A").dropna(thresh=3) # Read in the file and drop columns with 3 or more than 3 NaN values.
    nframes = dataframe.shape[0] / natoms # Counting the total number of frames in the trajectory.
    dataframe_list = np.array_split(dataframe, nframes) # Splitting the dataframe into chunks containing nframe atoms i.e basically into frames.
    zeros = np.zeros(natoms, dtype=int) # Zeros for initializing molecule array.
    for counter, frame in enumerate((dataframe_list)): # Iterating over the list containing frames.
        data = frame.reset_index() # reset index so as to start from zero again.
        data['index'] = np.arange(natoms) # Set the index column as atoms ranging from 0,1,2,3........natoms-1
        data.insert(1,'mols', zeros) # Insert the molecule array (only zeros right now)
        complete_traj[('').join(['time_',str(counter)])] = data # Make the dictionary key as time_0, time_1, time_2....etc. etc.
        #break
    return complete_traj, natoms # Return the completed traj.


def xyz_writer(input_traj, atoms=None, xyzname="output.xyz", com_traj=False):
    if os.path.exists(xyzname):
        os.remove(xyzname)
    if atoms is None:
        listdfs = list(input_traj.values())
        atoms = len(listdfs[0])
    if com_traj:
        atom_array = np.array(["A"]*atoms)
    else:
        atom_array = listdfs[0]['atoms'].to_numpy()
    str_array = np.zeros(atom_array.size, dtype=[('var1', 'U6'), ('var2', np.float64), ('var3', np.float64), ('var4', np.float64)])
    with open(xyzname, 'a') as file:
        for counter, key in enumerate((input_traj)):
            df = input_traj[key]
            file.write("{}\n".format(atoms))
            file.write("time = {}\n".format(counter))
            str_array['var1'] = atom_array
            str_array['var2'] = df['x'].to_numpy()
            str_array['var3'] = df['y'].to_numpy()
            str_array['var4'] = df['z'].to_numpy()
            np.savetxt(file, str_array, fmt='%5s %17.10f %17.10f %17.10f')
            #if counter==2:
            #    break
    return None




#ethane, natoms = read_aimd_pimd_xyz_faster("ethane.xyz")


def get_carbons(traj):
    out = {}
    for counter, time in enumerate((traj)):
        frame = traj[time].copy(deep=True)
        CC1 = frame[1080: (1080+4096)].copy(deep=True)
        CC2 = frame[6256: (6256+4096)].copy(deep=True)
        CC3 = frame[11432: (11432+4096)].copy(deep=True)
        ethane_frame = pd.concat([CC1, CC2, CC3], ignore_index=True)
        ethane_carbons = ethane_frame.loc[ethane_frame['atoms'] == 'C'].reset_index(drop=True)
        molecules = np.repeat(np.arange(0, len(ethane_carbons)/2, dtype=np.int16), 2)
        ethane_carbons['mols'] = molecules
        out[time] = ethane_carbons

    return out


#ethane_carbons = get_carbons(ethane)



def CC_vectors(traj, box=[33.4324, 35.3284, 131.6939]):
    out = {}
    box = np.array(box)
    for counter, time in enumerate((traj)):
        frame = traj[time].copy(deep=True)
        carbon_1st = (frame.groupby('mols').nth(0).reset_index())[['x', 'y', 'z']]
        carbon_2nd = (frame.groupby('mols').nth(1).reset_index())[['x', 'y', 'z']]
        r_ik =  (carbon_2nd - carbon_1st) - (box * ((carbon_2nd - carbon_1st)/box).round())  # mindist vectors NH's in this case
        normalisation = np.linalg.norm((r_ik[['x','y','z']].to_numpy()), axis=1, keepdims=True)
        r_ik = r_ik / normalisation
        r_ik['identity'] = np.arange(len(r_ik))
        out[time] = r_ik # Normalized C-C bond vectors with unique identifier coloumn
    return out


def substract_box(traj, box_lo=[-16.4952, -17.6852, -24.1489]):
    out = {}
    box_lo = np.array(box_lo)
    print("Please make sure the box_lo dimensions are", box_lo)
    for counter, time in enumerate((traj)):
        frame = traj[time].copy(deep=True)
        frame[['x', 'y', 'z']] =  frame[['x', 'y', 'z']] - box_lo
        out[time] = frame
    return out



#Carbons_from_zero = substract_box(ethane_carbons)

#carbon_vectors = CC_vectors(Carbons_from_zero, box=[33.4324, 35.3284, 131.6939])




def make_timeseries(traj):
    out = {}
    frame_list = list(traj.values())
    full_df = pd.concat(frame_list, ignore_index=True)
    grouped = full_df.groupby('identity')
    for name, group in grouped: # iterating over the grouped object we created.
        out[name] = group
    return out


#carbon_timeseries = make_timeseries(carbon_vectors)



def P2_Correlate(timeseries):
    p2 = 0
    for name in (timeseries):
        ts = timeseries[name]
        ts_x = ts['x'].to_numpy()
        ts_y = ts['y'].to_numpy()
        ts_z = ts['z'].to_numpy()
        p2 += direct_correlate(ts_x, ts_y, ts_z)
    P2_avg = p2 / len(timeseries)
    return P2_avg


#carbon_p2 = P2_Correlate(carbon_timeseries)


def master(xyz, file, save=False, box=[33.4324, 35.3284, 131.6939]):
    ethane, natoms = read_aimd_pimd_xyz_faster(xyz)
    ethane_carbons = get_carbons(ethane)
    Carbons_from_zero = substract_box(ethane_carbons)
    carbon_vectors = CC_vectors(Carbons_from_zero)
    carbon_timeseries = make_timeseries(carbon_vectors)
    carbon_p2 = P2_Correlate(carbon_timeseries)
    if save is True:
        np.savetxt(file, np.c_[carbon_p2.to_numpy()],fmt="%15.10f")
    return carbon_p2


if __name__ == '__main__':
    mix_p2 = master("mixture-3-7-nvt-prod2-94K.xyz", file="P2.dat", save=True)
