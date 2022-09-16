import numpy as np
import pandas as pd
from scipy import signal
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from numba import jit
import pickle
from modulefile import pickle_object, unpickle_object, read_aimd_pimd_xyz_faster, xyz_writer, master_sort, set_column_names

#pd.set_option("display.max_rows", None, "display.max_columns", None)

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

def numpy_correlate(array, cor_time=None):
    assert type(array) is np.ndarray,"Your code broke because of wrong inputs"
    c_t = np.correlate(array,array,mode='full')
    half = c_t.size//2
    c_t = c_t[half:] # half+max_time_for_corr_function+1] # what is nt ?
    c_t = c_t / np.linspace(nstep,nstep-nt,nt+1,dtype=np.float_) # nstep = np.linspace(len(array), len(array)-cor_time, cor_time+1, dtype=np.float_)
    return c_t

np.set_printoptions(suppress=True)
def fft_correlate(array):
    assert type(array) is np.ndarray,"Your code broke because of wrong inputs"
    pad_array = (np.pad(array, (0, len(array)), 'constant')).astype('complex128') # Padding zeroes and converting to complex
    fft = np.fft.fft(pad_array) # fast fourier (z)
    fft_conj = np.conjugate(fft) # complex conjugate (z*)
    sq_mod = fft*fft_conj # square modulus of complex number z is = zz*
    inv_fft = (np.fft.ifft(sq_mod)) #.astype('float') # inverse fft
    print(np.absolute(inv_fft)) # printing the norm of the resulting complex array since imaginary part is zero


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

def chunk_traj(file='fastsorted_file.txt', atoms=256):
    datatypes = {"index": np.int32, "mols" : np.int32 ,"atoms" : "str", "x" : np.float64, "y" : np.float64, "z" : np.float64}
    df_chunk = pd.read_csv(file, usecols=[0,1,2,3,4,5],  header=0, names=['index', 'mols', 'atoms', 'x', 'y', 'z'], chunksize=atoms, dtype=datatypes)
    return df_chunk

def read_sorted(File_parser):
    traj_dict = {}
    for counter, chunk in tqdm(enumerate(File_parser)):
        chunk.reset_index(drop=True, inplace=True)
        traj_dict[('').join(['time_',str(counter)])] = chunk
    return traj_dict

# Works for both NH3 and C2H2. Use this.
def rotcorr_input(data,atom_name = 'C', x=18.284, y=12.740, z=11.778, nh3_count=32):
    box = pd.DataFrame({"x":[x], "y":[y], "z":[z]})
    # corr_data = pd.DataFrame() # Blank dataframe for appending results.
    iterables = [np.linspace(1,nh3_count,nh3_count), ['NH1','NH2','NH3']]  # hardcoded. don't hardcode.
    index = pd.MultiIndex.from_product(iterables, names=['NH3-Mols', 'NH-vectors']) # multi-index indexing each nh3 molecule and three nh vectors per molecule.
    df_list = []
    for key in tqdm(data.keys()):
        #data[key] = data[key].astype({'index': 'int64', 'mols': 'int32', 'atoms': 'str', 'x': 'float64', 'y': 'float64', 'z': 'float64' })
        df = data[key]
        df1 = df.loc[df['atoms'] == atom_name] # N-dataframe
        df2 = df.loc[df['mols'].isin(df1['mols'])] # NH3-dataframe
        df2 = df2.groupby('mols').apply(lambda x: x.sort_values(['atoms'])).reset_index(drop=True)
        df1 = df1.drop_duplicates(subset=['mols']) # IF youare doing acetylene of similar molecules.
        df3 = df1.loc[df1.index.repeat(4)] # 4 is hardcoded #Repeat $ times frame of N
        df2 = df2[['x','y','z']] # NH3 dataframe
        df3 = df3[['x','y','z']] # N-repeated dataframe
        df2.reset_index(drop=True, inplace=True) # Resetting the indices for proper substraction
        df3.reset_index(drop=True, inplace=True)
        repeat = df2['x'].count()  # count to create box-dataframe of similar size for minimum image vector
        box_df = pd.concat([box]*repeat, ignore_index=True) # box-dataframe
        r_ik =  (df2 - df3) - (box_df * ((df2-df3)/box_df).round())  # mindist vectors NH's in this case
        r_ik = r_ik.loc[(r_ik!=0).any(axis=1)] #dropping zeros i.e N-N vectors
        normalisation = np.linalg.norm((r_ik[['x','y','z']].to_numpy()), axis=1, keepdims=True)
        r_ik = r_ik / normalisation # normalisation for removing the effects of bond vibrations
        r_ik = r_ik.set_index(index)
        #corr_data = corr_data.append(r_ik) # Appending is very slow - so appending to the list.
        df_list.append(r_ik)
        #break
    all_dfs = pd.concat(df_list)
    lexsorted_df = all_dfs.sort_index() # lex-sorting the dataframe at the end for efficient access. can be checked with df.index.is_lexsorted() = True or False
    return lexsorted_df

def do_corr(corr_data, nh3_count=32): #corr_data=corr_data
    corr_values = pd.DataFrame()
    k = 1 # dirty code
    for i in tqdm(range(1,nh3_count+1)): #harcoded
        NH1_x = scipy_correlate(corr_data.loc[(i,'NH1'),'x'].to_numpy())
        NH1_y = scipy_correlate(corr_data.loc[(i,'NH1'),'y'].to_numpy())
        NH1_z = scipy_correlate(corr_data.loc[(i,'NH1'),'z'].to_numpy())
        NH2_x = scipy_correlate(corr_data.loc[(i,'NH2'),'x'].to_numpy())
        NH2_y = scipy_correlate(corr_data.loc[(i,'NH2'),'y'].to_numpy())
        NH2_z = scipy_correlate(corr_data.loc[(i,'NH2'),'z'].to_numpy())
        NH3_x = scipy_correlate(corr_data.loc[(i,'NH3'),'x'].to_numpy())
        NH3_y = scipy_correlate(corr_data.loc[(i,'NH3'),'y'].to_numpy())
        NH3_z = scipy_correlate(corr_data.loc[(i,'NH3'),'z'].to_numpy())
        corr_values[f'column_{k}'] =   (NH1_x + NH2_x + NH3_x)/3.0
        corr_values[f'column_{k+1}'] = (NH1_y + NH2_y + NH3_y)/3.0  # Averaging over three vectors
        corr_values[f'column_{k+2}'] = (NH1_z + NH2_z + NH3_z)/3.0
        k += 3
    iterables = [map(lambda x: f'NH_{x}',np.linspace(1,nh3_count,nh3_count, dtype=int)), ['NH1','NH2','NH3']]  # hardcoded. don't hardcode.
    #%% iterables = [map(lambda x: f'NH_{x}',np.linspace(1,32,32, dtype=int)), ['x','y','z']]  # hardcoded. don't hardcode.
    index = pd.MultiIndex.from_product(iterables, names=['NH3-Mols', 'coordinates'])
    corr_values.set_axis(index, axis=1, inplace=True)
    corr_values = corr_values.sum(axis=1,level='coordinates')/float(index.get_level_values(level='NH3-Mols').nunique()) #Averaging over molecules
    corr_values = corr_values.sum(axis=1)/float(index.get_level_values(level='coordinates').nunique()) # Averaging over x,y,z
    return corr_values

def p2_corr(corr_data, nh3_count=32):
    corr_values = pd.DataFrame()
    k = 1
    for i in tqdm(range(1,nh3_count+1)): #33
        NH1 = direct_correlate(corr_data.loc[(i,'NH1'),'x'].to_numpy(), corr_data.loc[(i,'NH1'),'y'].to_numpy(), corr_data.loc[(i,'NH1'),'z'].to_numpy())
        NH2 = direct_correlate(corr_data.loc[(i,'NH2'),'x'].to_numpy(), corr_data.loc[(i,'NH2'),'y'].to_numpy(), corr_data.loc[(i,'NH2'),'z'].to_numpy())
        NH3 = direct_correlate(corr_data.loc[(i,'NH3'),'x'].to_numpy(), corr_data.loc[(i,'NH3'),'y'].to_numpy(), corr_data.loc[(i,'NH3'),'z'].to_numpy())
        corr_values[f'column_{k}'] =   NH1
        corr_values[f'column_{k+1}'] = NH2 # Averaging over three vectors
        corr_values[f'column_{k+2}'] = NH3
        k += 3
    iterables = [map(lambda x: f'NH_{x}',np.linspace(1,nh3_count,nh3_count, dtype=int)), ['NH1','NH2','NH3']]  # hardcoded. don't hardcode.
    index = pd.MultiIndex.from_product(iterables, names=['NH3-Mols', 'vectors'])
    corr_values.set_axis(index, axis=1, inplace=True)
    corr_values = corr_values.sum(axis=1,level='NH3-Mols') / float(index.get_level_values(level='vectors').nunique()) #Averaging over molecules
    print(float(index.get_level_values(level='vectors').nunique()))
    corr_values = corr_values.sum(axis=1) / float(index.get_level_values(level='NH3-Mols').nunique())
    return corr_values

if __name__ == '__main__':
    # Analysis for E&S Paper:
    print("Unpickling...")
    obj = unpickle_object("../Traj/90K/Final-90K/90K_full_sorted.pkl")
    AA_90k = set_column_names(obj)
    ##NH3 P2 Correlations
    print("NH3 Inputs...")
    NH3_rotcorr_input = rotcorr_input(AA_90k, atom_name = 'N', x=18.284, y=12.740, z=11.778, nh3_count=32)
    print("NH3 Correlations...")
    NH3_P2 = p2_corr(NH3_rotcorr_input, nh3_count=32)
    pickle_object(NH3_P2, "NH3/NH3_P2.pkl")
    NH3_time = np.arange(0, len(NH3_P2))
    np.savetxt("NH3/NH3_P2_90K.dat",np.c_[NH3_time, NH3_P2.to_numpy()],fmt="%15.10f")

    ## C2H2 P2 Correlations
    print("C2H2 Inputs...")
    C2H2_rotcorr_input = rotcorr_input(AA_90k, atom_name = 'C', x=18.284, y=12.740, z=11.778, nh3_count=32)
    print("C2H2 Correlations...")
    C2H2_P2 = p2_corr(C2H2_rotcorr_input, nh3_count=32)
    pickle_object(C2H2_P2, "C2H2/C2H2_P2.pkl")
    C2H2_time = np.arange(0, len(C2H2_P2))
    np.savetxt("C2H2/C2H2_P2_90K.dat",np.c_[C2H2_time, C2H2_P2.to_numpy()],fmt="%15.10f")
