import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from useful import read_aimd_pimd_xyz_faster, xyz_writer, pickle_object, unpickle_object
from ase.units import Hartree, Bohr # In ASE, defaults are eV and Ang so Hartree is a conv factor for eV and Bohr is conv factor for Ang.


def forces_units(force_traj, units=1.0):
    forces_units = {}
    for time in tqdm(force_traj):
        force_df = force_traj[time].copy(deep=True)
        force_df['x'] =  force_df['x'] * units
        force_df['y'] =  force_df['y'] * units
        force_df['z'] =  force_df['z'] * units
        forces_units[time] = force_df
    return forces_units

def read_energies(file, units=None):
    energy = pd.read_csv(file)
    energy_list = list(energy.to_numpy()[:, 1])
    if units is not None:
        energy_list = list(energy.to_numpy()[:, 1] * units)
    return energy_list

def write_pos_force_raw(traj, output):
    with open(output, 'a') as f:
        for counter, time in enumerate(tqdm(traj)):
            frame = traj[time].copy(deep=True)
            frame_xyz = frame[['x', 'y', 'z']].to_numpy().flatten()
            size = len(frame_xyz)
            np.savetxt(f, frame_xyz, fmt='%17.10f', newline='')
            f.write('\n')
            #if counter==2: break
    return None

def write_energy_raw(energy_list, output):
    energy_array = np.array(energy_list)
    with open(output, 'a') as f:
        np.savetxt(f, energy_array, fmt='%17.10f')
    return None

def write_staticbox_raw(box, output, nframe):
    box_matrix = np.identity(3)
    box_matrix[0, 0] = box[0]
    box_matrix[1, 1] = box[1]
    box_matrix[2, 2] = box[2]
    box_raw = box_matrix.flatten()
    with open(output, 'a') as f:
        for line in range(nframe):
            np.savetxt(f, box_raw, fmt='%17.10f', newline='')
            f.write('\n')
    return None


def write_type_raw(pos_traj, map_dict, output):
    frame = pos_traj['time_0'].copy(deep=True)
    types = frame['atoms'].map(map_dict).to_numpy()
    assert (np.isnan(np.sum(types)) == False), 'Hey..!! Your mapping dictionary is incomplete'
    type_unique = np.unique(types)
    map_dict_reverse = dict([(value, key) for key, value in map_dict.items()])
    type_map = np.array([map_dict_reverse[k] for k in type_unique])
    with open((output+'.raw'), 'w') as f:
        np.savetxt(f, types, fmt='%-5d', newline='')
    with open((output+'_map.raw'), 'w') as f:
        np.savetxt(f, type_map, fmt='%-5s', newline='')
    return None



if __name__ == '__main__':
    pos, _ = read_aimd_pimd_xyz_faster("out/Pos-200K-NN.xyz")
    force, _ = read_aimd_pimd_xyz_faster("out/Force-200K-NN.xyz")

    conversion = Hartree/Bohr
    force_eVperA = forces_units(force, units=conversion)
    write_pos_force_raw(force_eVperA, 'Raw/force.raw')

    Energy_eV = read_energies("out/En-200K-NN.dat", units=Hartree)
    write_energy_raw(Energy_eV, 'Raw/energy.raw')

    write_pos_force_raw(pos, 'Raw/coord.raw')

    write_type_raw(pos, {'H': 0, 'C': 1, 'N': 2}, 'Raw/type')

    write_staticbox_raw(box=[18.284, 12.740, 11.778], output='Raw/box.raw', nframe=len(pos))
