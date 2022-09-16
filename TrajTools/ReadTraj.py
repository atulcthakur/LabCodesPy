import time
#import re
import math
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def pickle_object(obj, file):
    pickle.dump(obj, open(file, "wb" ))

def unpickle_object(file):
    obj = pickle.load(open(file, "rb" ))
    return obj


# # Functions Reading the Trajectory


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
        if counter == 1:
            break
    return complete_traj


# In[5]:


def read_normal_pimd_xyz(xyzname='All_traj.xyz', atoms=256):
    chunksize_ = atoms + 2 #because there are two comment lines.
    df_chunk = pd.read_csv(xyzname, usecols=[0,1,2,3], delim_whitespace=True, names=['atoms','x','y','z'], chunksize=chunksize_)
    complete_traj = {}
    for counter,chunk in tqdm(enumerate(df_chunk)):
        frame = chunk.copy(deep=True)
        frame = frame.dropna()
        frame = frame.loc[frame['atoms']!='#'] #Quick fix for PIMD Trajs but not a good one
        #frame = frame.loc[frame['z']!='time']
        frame.reset_index(inplace=True)
        frame['index'] = np.arange(atoms)
        frame.insert(1,'mols', np.zeros(atoms, dtype=int))   # 256 with atoms
        frame = frame.astype({"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64})
        complete_traj[('').join(['time_',str(counter)])] = frame
        #if counter == 1:
        #    break
    return complete_traj


def read_duplicate_corrected_traj(xyzname, atoms=256):
    datatypes = {"atoms" : "str", "X" : np.float64, "Y" : np.float64, "Z" : np.float64}
    xyz_chunk = pd.read_csv(xyzname, delim_whitespace=False, dtype=datatypes, chunksize=atoms)
    output = {}
    for counter,chunk in tqdm(enumerate(xyz_chunk)):
        chunk.reset_index(inplace=True)
        chunk['index'] = np.arange(atoms)
        chunk.insert(1,'mols', np.zeros(atoms, dtype=int))
        output[('').join(['time_',str(counter)])] = chunk
        #if counter == 10:
        #    break
    return output


def read_traj(total_frames=5,atoms=256,file="AA-pos-1.xyz"):
    frames_dict = {}
    #total_frame = re.findall(r"^\w+",'time')
    #print(total_frame)
    frames=0
    var=2
    #df = pd.read_csv("AA-pos-1.xyz", nrows=256, skiprows=var ,usecols=[0,1,2,3], delim_whitespace=True, names=['atoms','x','y','z'])
    while (frames < total_frames):
        df = pd.read_csv(file, nrows=atoms, skiprows=var ,usecols=[0,1,2,3], delim_whitespace=True, names=['atoms','x','y','z'], dtype = {"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64})
        #column data typesTHINK!!!!
        #df = pd.read_csv("joint.txt", nrows=24, skiprows=var ,usecols=[0,1,2,3], delim_whitespace=True, names=['atoms','x','y','z'])
        #df['atoms'] = pd.Categorical(df['atoms'], ["C", "N", "H"])
        #df = df.sort_values(by='atoms')
        #df = df.reset_index(drop=True)
        df.reset_index(level=0, inplace=True)
        df.insert(1,'mols', np.zeros(atoms, dtype=int))
        frames_dict[('').join(['time_',str(frames)])] = df
        frames += 1
        var += atoms+2#258
    return frames_dict



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


# # Function to Fast-Sort and Write the Trajectory:

# In[7]:


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


def write_sorted(sorted_traj_dict ,filename='sorted_file.txt'):
    file = open(filename, 'w')
    file.close()
    for sorted_frames in sorted_traj_dict.keys():
        current_df = sorted_traj_dict[sorted_frames]
        current_df.to_csv(filename, header=False, index=False, mode='a')
    return None




# Sorting the 90K Traj
duplicate_removed_traj_90k = read_duplicate_corrected_traj("90K_duplicate_removed.xyz", atoms=256)
one_frame = {"time_0": duplicate_removed_traj_90k["time_0"] }
oneframe_sorted = sort_traj(one_frame, atoms=256, x=18.284, y=12.740, z=11.778, cutoff=1.5)
full_sorted_90k =  map_index(duplicate_removed_traj_90k, oneframe_sorted["time_0"], write_output=False)
pickle_object(full_sorted_90k, "90K_full_sorted.pkl")
write_sorted(full_sorted_90k, "90K_full_sorted.xyz")
