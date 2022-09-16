import numpy as np
import pandas as pd
import re
import math
from tqdm import tqdm



def frames_list(filename="AA-90K-MD-pos-1.xyz", atoms=256):
    xyz = pd.read_csv(filename, names=['atoms','X','Y','Z'], delim_whitespace=True, comment="i")
    all_frame_df = xyz.dropna(axis=0, how="any").reset_index(drop=True)
    datatypes = {"atoms" : "str", "X" : np.float64, "Y" : np.float64, "Z" : np.float64}
    all_frame_df = all_frame_df.astype(datatypes)
    list_df = [all_frame_df[i:i+atoms] for i in tqdm(range(0,all_frame_df.shape[0],atoms))]
    return list_df





def extract_times(filename="AA-90K-MD-pos-1.xyz"):
    timelist = []
    file = open(filename, "r")
    for line in tqdm(file):
        if re.search("time", line):
            time = int((line.split()[5]).split(".")[0])
            timelist.append(time)
    return timelist




def remove_duplicates(list_df, timelist, output_name="complete.xyz"):
    assert (len(list_df)==len(timelist)),"Something Went Terribly Wrong Somewhere! Check Your Inputs"
    corrected_xyzs = {}
    for count, time in tqdm(enumerate(timelist)):
        corrected_xyzs[time] = list_df[count]
    complete_xyz = pd.concat(list(corrected_xyzs.values()), ignore_index=True)
    complete_xyz.to_csv(output_name, index=False)
    return None #, corrected_xyzs

if __name__ == '1st':
    listdf = frames_list(filename="Pos_90K.xyz", atoms=256)
    timelist = extract_times(filename="Pos_90K.xyz")
    remove_duplicates(listdf, timelist, output_name="90K_duplicate_removed.xyz" )

if __name__ == '__main__':
    listdf = frames_list(filename="Vel_90K.xyz", atoms=256)
    timelist = extract_times(filename="Vel_90K.xyz")
    remove_duplicates(listdf, timelist, output_name="Vel_90K_duplicate_removed.csv" )
