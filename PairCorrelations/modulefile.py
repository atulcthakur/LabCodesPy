import numpy as np
import pandas as pd 
import math
from tqdm.notebook import tqdm

def chunk_traj(file='fastsorted_file.txt', atoms=256):
    datatypes = {"index": np.int32, "mols" : np.int32 ,"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64}
    df_chunk = pd.read_csv(file, usecols=[0,1,2,3,4,5], header=0, names=['index', 'mols', 'atoms', 'x', 'y', 'z'], chunksize=atoms, dtype=datatypes)
    return df_chunk

def read_sorted(File_parser):
    traj_dict = {}
    for counter, chunk in enumerate(tqdm(File_parser)):
        chunk.reset_index(drop=True, inplace=True)
        traj_dict[('').join(['time_',str(counter)])] = chunk
    return traj_dict