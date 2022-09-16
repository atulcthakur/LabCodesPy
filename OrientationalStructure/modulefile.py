import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import os
import pickle

def pickle_object(obj, file):
    pickle.dump(obj, open(file, "wb" ))

def unpickle_object(file):
    obj = pickle.load(open(file, "rb" ))
    return obj

def read_aimd_pimd_xyz_faster(filename, natoms=None, PIMD=False):
    "Read AIMD/PIMD xyz Faster"
    print(f"Reading File {filename}") # Printing the filename
    if natoms is None: # If natoms is not set by the user then read the file to get natoms.
        with open(filename, 'r') as f:  # opening the file.
            natoms = int(f.readline().strip()) # reading in natoms.
    complete_traj = {}  # Final traj to store the output.
    datatypes = {"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64} # Datatypes that we're gonna cast on the colums read from the file.
    if PIMD: # If the trajectory is from i-pi
        comment_symbol = "#"  # set the comment symbol as #.
    else:
        comment_symbol = "i" # else use the following for normal AIMD cp2k trajectories. (Maybe different for Lammps xyz files.)
    dataframe = pd.read_csv(filename, usecols=[0,1,2,3], delim_whitespace=True, names=['atoms','x','y','z'], comment=comment_symbol, dtype=datatypes).dropna(thresh=3) # Read in the file and drop columns with 3 or more than 3 NaN values.
    nframes = dataframe.shape[0] / natoms # Counting the total number of frames in the trajectory.
    dataframe_list = np.array_split(dataframe, nframes) # Splitting the dataframe into chunks containing nframe atoms i.e basically into frames.
    zeros = np.zeros(natoms, dtype=int) # Zeros for initializing molecule array.
    for counter, frame in enumerate(tqdm(dataframe_list)): # Iterating over the list containing frames.
        data = frame.reset_index() # reset index so as to start from zero again.
        data['index'] = np.arange(natoms) # Set the index column as atoms ranging from 0,1,2,3........natoms-1
        data.insert(1,'mols', zeros) # Insert the molecule array (only zeros right now)
        complete_traj[('').join(['time_',str(counter)])] = data # Make the dictionary key as time_0, time_1, time_2....etc. etc.
        #break
    return complete_traj, natoms # Return the completed traj.


def chunk_traj(file='fastsorted_file.txt', atoms=256):
    datatypes = {"index": np.int32, "mols" : np.int32 ,"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64}
    df_chunk = pd.read_csv(file, usecols=[0,1,2,3,4,5], header=0, names=['index', 'mols', 'atoms', 'x', 'y', 'z'], chunksize=atoms, dtype=datatypes)
    return df_chunk

def read_sorted(File_parser):
    traj_dict = {}
    for counter, chunk in enumerate(tqdm(File_parser)):
        chunk.reset_index(drop=True, inplace=True)
        traj_dict[('').join(['time_',str(counter)])] = chunk
        if counter == 1000: break
    return traj_dict

# COPY PASTED FROM RADIAL_DISTRIBUTION.IPYNB
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
        for counter, key in enumerate(tqdm(input_traj)):
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

# COPY PASTED FROM RADIAL_DISTRIBUTION.IPYNB
def make_whole(sorted_traj, atoms=256, x=18.284, y=12.740, z=11.778, write_traj=False):
    x_half = x/2.0
    y_half = y/2.0
    z_half = z/2.0
    output = {}
    for counter, key in enumerate(tqdm(sorted_traj)):
        df = sorted_traj[key]
        wholed_df = df.copy(deep=True)
        select_df = df[['index', 'x', 'y', 'z']]
        grouped_first_entry = df.groupby("mols").nth(0)
        repeat_count = df.groupby("mols").size()
        ref_df = grouped_first_entry.loc[grouped_first_entry.index.repeat(repeat_count)].reset_index(drop=True)
        ref_df = ref_df[['index', 'x', 'y', 'z']]
        r_ix = select_df - ref_df
        conditions_x, values_x = [(r_ix['x'] > x_half), (r_ix['x'] < -x_half)], [-x, x]
        conditions_y, values_y = [(r_ix['y'] > y_half), (r_ix['y'] < -y_half)], [-y, y]
        conditions_z, values_z = [(r_ix['z'] > z_half), (r_ix['z'] < -z_half)], [-z, z]
        shifts_x = np.select(conditions_x, values_x, default=0.0)
        shifts_y = np.select(conditions_y, values_y, default=0.0)
        shifts_z = np.select(conditions_z, values_z, default=0.0)
        wholed_df['x'] =  df['x'] + shifts_x
        wholed_df['y'] =  df['y'] + shifts_y
        wholed_df['z'] =  df['z'] + shifts_z
        output[key] = wholed_df
    if write_traj:
        xyz_writer(output, xyzname="wholed_traj.xyz")
        return None
    return output

def read_normal_xyz(xyzname='All_traj.xyz', atoms=256):
    chunksize_ = atoms + 2 #because there are two comment lines.
    df_chunk = pd.read_csv(xyzname, usecols=[0,1,2,3], delim_whitespace=True, names=['atoms','x','y','z'], chunksize=chunksize_)
    complete_traj = {}
    for counter,chunk in tqdm(enumerate(df_chunk)):
        frame = chunk.copy(deep=True)
        frame = frame.dropna()
        #frame = frame.loc[frame['atoms']!='#'] #Quick fix for PIMD Trajs but not a good one
        #print(frame)
        frame = frame.loc[frame['z']!='time']
        frame.reset_index(inplace=True)
        frame['index'] = np.arange(atoms)
        frame.insert(1,'mols', np.zeros(atoms, dtype=int))   # 256 with atoms
        frame = frame.astype({"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64})
        complete_traj[('').join(['time_',str(counter)])] = frame
        if counter == 10:
            break
    return complete_traj

# TESTING GROUND
def sort_traj(frames_dict, total_frames=5, atoms=256, x=18.284, y=12.740, z=11.778, cutoff=1.8, filename='foo.txt'):
    box = np.array([x, y, z])
    output_dict = {}
    for key in tqdm(frames_dict.keys()):
        frame_numpy = frames_dict[key].to_numpy()  # Here's its converting everything to objects.
        molecule_array =  np.zeros([atoms])
        index_set = set()
        k,m,increment = 1,0,0
        index_set.add(frame_numpy[0,0]) # adding 1st element
        for i in range(0,len(frame_numpy)):
            value1 = frame_numpy[i,0]
            count = 0
            if (value1 not in index_set):
                increment += 1
            while(k <= (len(frame_numpy)-1)):
                x = np.array([frame_numpy[i,3],frame_numpy[i,4],frame_numpy[i,5]]).astype('float64')
                y = np.array([frame_numpy[k,3],frame_numpy[k,4],frame_numpy[k,5]]).astype('float64')
                r_ik = (x-y) - (box * np.around((x-y)/box)) # you can safely remove .astype('float_') since the dtypes are being casted above- Removed
                distance = np.linalg.norm(r_ik)
                value = frame_numpy[k,0]  # we're no where comparing value and value1, so the dtypes can be objects - Not a big problem
                if (k > i and distance < cutoff):
                    frame_numpy[[m+1, k]] = frame_numpy[[k, m+1]]
                    index_set.add(value), index_set.add(value1)
                    count += 1
                    m += 1
                elif (i==k and value1 not in index_set):
                    m += 1
                k += 1
            frame_numpy[i,1] = increment # so since i end at len -2 last two will be voilated
            k = m+1
        sorted_dataframe = pd.DataFrame(frame_numpy, columns=['index','mols','atoms','x','y','z'])
        output_dict[key] = sorted_dataframe      #frame_numpy
    #np.savetxt(filename, frame_numpy, delimiter="  ", fmt='%s')
    #np.savetxt('mols.txt', molecule_array, delimiter="  ", fmt='%s')
    return output_dict

# COPY PASTED FROM HB-Dynamics-MDAnalysis.IPYNB
def add_virtual_site(structure, molecules=["C","H"], masses={"C": 12, "H": 1, "N": 14}):
    out = {}
    for frame in tqdm(structure):
        df = structure[frame].copy(deep=True)
        CC_df = (df.loc[(df['atoms'] == molecules[0])]).reset_index(drop=True) # Dataframe of carbon atoms
        CC_df['mass'] = CC_df['atoms'].map(masses) # Map the masses to respective atoms as given in specified mass dictionary (input)
        assert (not CC_df['mass'].isnull().any()), "Some of the masses have not been assigned which means your mass dictinary is incomplete. Please provide mass list of all unique atoms in the trajectory."
        #### Here first multiply by mass column and then do the following.
        CC_df[['x','y','z']] = CC_df[['x','y','z']].mul(CC_df['mass'], axis=0) #  Multiplying by the masses
        com_df = CC_df.groupby("mols").sum() # NumerCC_dfor of the COM formula i.e x1*m1 + x2*m2 + x3*m3..... and same along y and z for every molecule.
        com_df[['x','y','z']]  = com_df[['x','y','z']].div(com_df['mass'], axis=0) # Dividing by the total mass to get x_cm, y_cm and z_cm
        #com_df = com_df.reset_index(drop=True)
        com_df['atoms'] = np.array(['X']*len(com_df))
        com_df['index'] = np.arange(len(df), len(df)+len(com_df))
        com_df['mols'] = np.unique(CC_df['mols'])
        out[frame] = pd.concat([df, com_df.drop(columns=['mass'])]).sort_values(by=['mols'], ignore_index=True)
    return out


def map_index(tosort, sorted_df, write_output=False, filename="fastsorted_file.txt"): # e.g. tosort=full_dictionary_that is to be sorted
    sorted_index = sorted_df['index'] # index using which we want to sort every dataframe in the trajectory
    fastsorted = {} # create a dictionary to append results to.
    for key in tqdm(tosort.keys()):
        curr_df = tosort[key] # current dataframe
        curr_df = curr_df.reindex(sorted_index) #rearrange the dataframe according to sorted index
        curr_df.reset_index(drop=True, inplace=True) # now reset the index back to normal
        curr_df['mols'] = sorted_df['mols'] # now set the mols column the same as in the reference df
        fastsorted[key] = curr_df # Append the modified dataframe to the new dictionary
    if write_output:
        list_dfs = list(fastsorted.values())
        (pd.concat(list_dfs)).to_csv(filename, index=False)
    return fastsorted
