import numpy as np
import pandas as pd
import math
import scipy as sp
from tqdm.notebook import tqdm
from numba import jit
from matplotlib import pyplot as plt
import os
import warnings
from useful import read_aimd_pimd_xyz_faster, pickle_object, unpickle_object, xyz_writer, make_whole, sort_traj, map_index
warnings.filterwarnings(action='ignore')
from MDAnalysis.topology.XYZParser import XYZParser as read
from MDAnalysis.topology.guessers import guess_bonds, guess_angles, guess_dihedrals, guess_improper_dihedrals
from MDAnalysis.transformations import wrap, unwrap
import MDAnalysis as mda

#mda.__version__

def master_sort(xyzname=None, natoms=None, box=[18.284, 12.740, 11.778]):
    """
    Takes in the AIMD trajectory object, sorts it and returns the sorted trajectory.
    """
    oneframe = {"time_0": xyzname["time_0"]}  #First frame of the xyz trajectory.
    if natoms is None:
        natoms = len(oneframe["time_0"])
    sorted_oneframe = sort_traj(oneframe, atoms=natoms, x=box[0], y=box[1], z=box[2], cutoff=1.5) # Sorts out the first frame in the trajectory.
    #print(sorted_oneframe)
    sorted_xyz = map_index(xyzname, sorted_df=sorted_oneframe["time_0"]) # Sorts out the reamining frames.
    return sorted_xyz

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

mass = {"C": 12, "H": 1, "N": 14}
def add_virtual_site_mod(structure, molecules=["C","H"], masses={"C": 12, "H": 1, "N": 14}):
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
        C2H2_df = df.loc[(df['mols'].isin(CC_df['mols']))] # C2H2 DF
        C2H2_df['atoms'].replace('H', 'Z', inplace=True)
        NH3_df = df.loc[~(df['mols'].isin(CC_df['mols']))] # NH3 DF
        out_frame = pd.concat([C2H2_df, NH3_df, com_df.drop(columns=['mass'])]).sort_values(by=['mols'], ignore_index=True)
        out_frame = out_frame.groupby('mols').apply(lambda x: x.sort_values(['atoms'])).reset_index(drop=True)
        out[frame] = out_frame
    return out

def make_top(xyzfile, del_bonds=None, print_types=True, vdw_radii={'X': 1.1, 'Z': 1.1}, unit_cell=None, wrap_atoms=False):
    """ Assembling Top with a "X" center
        del_bonds = ('C', 'C') for example.
        unit_cell should have box length and angles, for example
        unit_cell = [18.284, 25.48, 23.556, 90, 90, 90]
    """
    top = read(xyzfile)
    verse = mda.Universe(top.parse(), xyzfile)
    all_bonds = guess_bonds(verse.atoms, verse.coord.positions, vdwradii=vdw_radii)
    verse.add_TopologyAttr('bonds', all_bonds)
    if del_bonds is not None:
        delete_bonds_list = verse.bonds.select_bonds(del_bonds).to_indices()
        verse.delete_bonds(delete_bonds_list)
    all_angles = guess_angles(verse.bonds)
    verse.add_TopologyAttr('angles', all_angles)
    all_dih = guess_dihedrals(verse.angles)
    verse.add_TopologyAttr('dihedrals', all_dih)
    all_imp = guess_improper_dihedrals(verse.angles)
    verse.add_TopologyAttr('impropers', all_imp)
    if wrap_atoms is True and unit_cell is not None:
        verse.dimensions = np.array(unit_cell)
        ag = verse.atoms
        transform = wrap(ag)
        verse.trajectory.add_transformations(transform)
    if print_types:
        print("The bond Types are", verse.bonds.types())
        print("The angle Types are", verse.angles.types())
        print("The dihedral Types are", verse.dihedrals.types())
        print("The improper Types are", verse.impropers.types())
    return verse

def header(data_file, mda_universe, dihedrals=False, impropers=False, box=None):
    """
    box = [xlo, xhi, ylo, yhi, zlo, zhi]
    """
    with open(data_file, 'w') as lammpsdata:
        lammpsdata.write("LAMMPS DATA File\n")
        lammpsdata.write("\n")
        lammpsdata.write(f"{mda_universe.atoms.n_atoms} atoms\n")
        lammpsdata.write(f"{len(mda_universe.bonds)} bonds\n")
        lammpsdata.write(f"{len(mda_universe.angles)} angles\n")
        if dihedrals:
            lammpsdata.write(f"{len(mda_universe.dihedrals)} dihedrals\n")
        if impropers:
            lammpsdata.write(f"{len(mda_universe.impropers)} impropers\n")
        lammpsdata.write("\n")
        lammpsdata.write(f"{np.unique(mda_universe.atoms.types).shape[-1]} atom types\n")
        lammpsdata.write(f"{len(mda_universe.bonds.types())} bond types\n")
        lammpsdata.write(f"{len(mda_universe.angles.types())} angle types\n")
        if dihedrals:
            lammpsdata.write(f"{len(mda_universe.dihedrals.types())} dihedral types\n")
        if impropers:
            lammpsdata.write(f"{len(mda_universe.impropers.types())} improper types\n")
        if box is not None:
            lammpsdata.write("\n")
            lammpsdata.write(f"{box[0]} {box[1]} xlo xhi\n")
            lammpsdata.write(f"{box[2]} {box[3]} ylo yhi\n")
            lammpsdata.write(f"{box[4]} {box[5]} zlo zhi\n")

def write_masses(data_file, mass_dict=None):
    """ mass_dict = {C: 12, H: 1, X: 1, N: 14, Z: 1}  """
    if mass_dict is None:
        return "Please provide dictionary of masses."
    with open(data_file, 'a') as lammpsdata:
        lammpsdata.write("\n")
        lammpsdata.write("Masses\n")
        lammpsdata.write("\n")
        for index, atom in enumerate(mass_dict, start=1):
            lammpsdata.write(f"{index:6d} {mass_dict[atom]:4.4f}\n")

def write_atoms(data_file, mda_universe, mass_dict=None, charge_dict=None):
    """
   charge_dict = {C: 1, H: 2, X: 3, N: 4, Z: 5}
    """
    with open(data_file, 'a') as lammpsdata:
        lammpsdata.write("\n")
        lammpsdata.write("Atoms\n")
        lammpsdata.write("\n")
        mass_list = list(mass_dict)
        mapping = [(mass_list.index(atom)+1) for atom in mda_universe.atoms.names]
        charge_mapping = [charge_dict[atom] for atom in mda_universe.atoms.names]
        molecule_id_list = 0
        for index, atomid in enumerate(mapping):
            lammpsdata.write(f"{index+1:6d}  {molecule_id_list:6d}  {atomid:6d}  {charge_mapping[index]:15.6f}  {mda_universe.atoms.positions[index, 0]:15.8f}  {mda_universe.atoms.positions[index, 1]:15.8f}  {mda_universe.atoms.positions[index, 2]:15.8f}\n")

def write_bonds(data_file, mda_universe):
    """
    """
    with open(data_file, 'a') as lammpsdata:
        lammpsdata.write("\n")
        lammpsdata.write("Bonds\n")
        lammpsdata.write("\n")
        sr_no = np.arange(len(mda_universe.bonds.to_indices())) + 1
        value = 0
        bondtype_mapping = {bondtype:index for index, bondtype in enumerate(mda_universe.bonds.types(), start=1)}
        for btype, bondid in bondtype_mapping.items():
            blist = mda_universe.bonds.select_bonds(btype).dump_contents() + 1
            for index, atoms in enumerate(blist):
                lammpsdata.write(f"{sr_no[index+value]:6d}  {bondid:6d}  {blist[index, 0]:6d}  {blist[index, 1]:6d}\n")
            value += index + 1


def write_angles(data_file, mda_universe, n=6):
    """
    """
    with open(data_file, 'a') as lammpsdata:
        lammpsdata.write("\n")
        lammpsdata.write("Angles\n")
        lammpsdata.write("\n")
        sr_no = np.arange(len(mda_universe.angles.to_indices())) + 1
        value = 0
        angletype_mapping = {angletype:index for index, angletype in enumerate(mda_universe.angles.types(), start=1)}
        for atype, angleid in angletype_mapping.items():
            alist = mda_universe.angles.select_bonds(atype).dump_contents() + 1
            for index, atoms in enumerate(alist):
                lammpsdata.write(f"{sr_no[index+value]:6d}  {angleid:6d}  {alist[index, 0]:6d}  {alist[index, 1]:6d}  {alist[index, 2]:6d}\n")
            value += index + 1


def write_dihedrals(data_file, mda_universe):
    """
    """
    with open(data_file, 'a') as lammpsdata:
        lammpsdata.write("\n")
        lammpsdata.write("Dihedrals\n")
        lammpsdata.write("\n")
        sr_no = np.arange(len(mda_universe.dihedrals.to_indices())) + 1
        value = 0
        dihedraltype_mapping = {dihedraltype:index for index, dihedraltype in enumerate(mda_universe.dihedrals.types(), start=1)}
        for dtype, dihedralid in dihedraltype_mapping.items():
            dlist = mda_universe.dihedrals.select_bonds(dtype).dump_contents() + 1
            for index, atoms in enumerate(dlist):
                lammpsdata.write(f"{sr_no[index+value]:6d}  {dihedralid:6d}  {dlist[index, 0]:6d}  {dlist[index, 1]:6d}  {dlist[index, 2]:6d}  {dlist[index, 3]:6d}\n")
            value += index + 1

def write_impropers(data_file, mda_universe):
    """
    """
    with open(data_file, 'a') as lammpsdata:
        lammpsdata.write("\n")
        lammpsdata.write("Impropers\n")
        lammpsdata.write("\n")
        sr_no = np.arange(len(mda_universe.impropers.to_indices())) + 1
        value = 0
        impropertype_mapping = {impropertype:index for index, impropertype in enumerate(mda_universe.impropers.types(), start=1)}
        for itype, improperid in impropertype_mapping.items():
            ilist = mda_universe.impropers.select_bonds(itype).dump_contents() + 1
            for index, atoms in enumerate(ilist):
                lammpsdata.write(f"{sr_no[index+value]:6d}  {improperid:6d}  {ilist[index, 0]:6d}  {ilist[index, 1]:6d}  {ilist[index, 2]:6d}  {ilist[index, 3]:6d}\n")
            value += index + 1

def write_lammps_data(data_file, mda_universe , dihedrals=True, impropers=True, box=[0,1, 0, 1, 0, 1], mass_dict={'C': 12, 'H': 1, 'X': 1, 'N': 14, 'Z': 1}, charge_dict = {'C': 1, 'H': 2, 'X': 3, 'N': 4, 'Z': 5}):
    """

    """
    header(data_file, mda_universe, dihedrals, impropers, box)
    write_masses(data_file, mass_dict)
    write_atoms(data_file, mda_universe, mass_dict, charge_dict)
    write_bonds(data_file, mda_universe)
    write_angles(data_file, mda_universe)
    if dihedrals:
        write_dihedrals(data_file, mda_universe)
    if impropers:
        write_impropers(data_file, mda_universe)
    return None



### Using the AIMD 90K PBED3 configuration for DeepMD.
#traj,na = read_aimd_pimd_xyz_faster("one_frame.xyz")
#traj = set_column_names(traj)
#strct = master_sort(xyzname=traj, natoms=None, box=[18.284, 12.740, 11.778])
#whole = make_whole(strct, atoms=256, x=18.284, y=12.740, z=11.778, write_traj=True)
#crystal = make_top('one_frame.xyz', wrap_atoms=False, unit_cell=[18.284, (25.48)/2, (23.556)/2, 90, 90, 90])
#write_lammps_data('AA-deepmd-3rd.data', crystal, impropers=False, dihedrals=False, box=[0, 18.284, 0, (25.48)/2, 0, (23.556)/2], mass_dict={'N': 14.0067, 'H': 1.00784, 'C': 12.0107},\
#                  charge_dict = {'N': 0.00 , 'H': 0.00, 'C': 0.00})


if __name__ == '__main__':
    xyz, natoms = read_aimd_pimd_xyz_faster("Pos-90K-NN.xyz")
    one_frame = {'time_0': xyz['time_0']}
    xyz_writer(input_traj=one_frame, xyzname='one_frame.xyz')
    crystal = make_top('one_frame.xyz', wrap_atoms=False, unit_cell=[18.284, (25.48)/2, (23.556)/2, 90, 90, 90])
    write_lammps_data('Deepmd.data', crystal, impropers=False, dihedrals=False, box=[0, 18.284, 0, (25.48)/2, 0, (23.556)/2], mass_dict={'H': 1.00784, 'C': 12.0107, 'N': 14.0067},\
    charge_dict = {'H': 0.00, 'C': 0.00, 'N': 0.00})
