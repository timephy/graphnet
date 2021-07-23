import numpy as np
import logging
from numpy.lib.format import open_memmap
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def cart2polar(x, y, z):
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    return phi, theta

def polar2cart(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def get_energies(tree_data, tree_idx):
    pdg = tree_data['particle']['pdg_encoding']
    energies = tree_data['particle']['energy']
    neutrino_energies = energies[tree_idx['start']]
    nevents = len(tree_idx)
    track_energies = np.zeros(nevents)
    track_lengths = np.zeros(nevents)
    shower_energies = np.zeros(nevents)

    invis_mask = (np.abs(pdg) == 12) | (np.abs(pdg) == 14) | (np.abs(pdg) == 16)
    invis_mask[tree_idx['start']] = False
    e_invis = [np.sum(energies[start:stop][invis_mask[start:stop]])
               for start, stop in tqdm(tree_idx, desc='Getting invisible energies')]
    e_invis = np.array(e_invis)

    track_mask = np.abs(pdg) == 13
    n_muons = np.array([np.sum(track_mask[start:stop])
                        for start, stop in tqdm(tree_idx, desc='Getting track events')])
    has_track = np.where(n_muons>0)[0]
    # The main muon is the muon with the highest energy in each event. We assume that the track energy and length come mostly from it
    get_main_muon = lambda start, stop: np.argmax((energies[start:stop])[track_mask[start:stop]])
    # This gets the indices of all main muons over all events
    muon_idx = np.array([np.arange(start, stop, dtype=int)[track_mask[start:stop]][get_main_muon(start, stop)]
                for start, stop in tqdm(tree_idx[has_track], desc='Getting muon indices')])
    track_energies[has_track] = energies[muon_idx]
    track_lengths[has_track] = tree_data['particle']['length'][muon_idx]
    shower_energies = neutrino_energies - (track_energies + e_invis)
    df = pd.DataFrame(np.vstack([neutrino_energies, track_energies, shower_energies, track_lengths]).T,
                      columns=['neutrino_energy', 'track_energy', 'shower_energy', 'track_length'])
    return df

def load_primary_information(mcprimary):
    # Have to split this in two parts because of hierarchical structure of input data
    # maybe there's a smarter way of doing this
    labels = ['pdg_encoding', 'time', 'energy']
    superlabels = ['pos', 'dir']
    df_0 = pd.DataFrame(mcprimary[labels])
    df_1 = pd.concat([pd.DataFrame(mcprimary[l]) for l in superlabels], axis=1)
    return pd.concat([df_0, df_1], axis=1)

def get_truths(indir, eps=1e-3, force_recalculate=False, save_energies=False):
    """
    Get all truth information necessary

    Parameters:
    ----------
    indir: str
        Input directory from i3cols output
    eps: float, optional
        Value to offset for 0-valued track energies
        Otherwise NaN with np.log
    """
    mctree_data = open_memmap(os.path.join(indir, 'I3MCTree', 'data.npy'))
    mctree_idx = np.load(os.path.join(indir, 'I3MCTree', 'index.npy'))

    primary_idx = mctree_idx['start']
    mcprimary = mctree_data[primary_idx]['particle']

    primaries = load_primary_information(mcprimary)
    try:
        if force_recalculate:
            logging.info('Recalculating cascade and track energy information')
            raise FileNotFoundError  # hacky lol
        energies = pd.read_pickle(os.path.join(indir, 'processed', 'energies.pkl'))
        logging.info('Loading precalculated cascade and track energy information')
        if save_energies:
            logging.warning('Processed cascade and track energy information found, not saving to file')
    except FileNotFoundError:
        energies = get_energies(mctree_data, mctree_idx)
        if save_energies:
            logging.info('Saving cascade and track information')
            Path(os.path.join(indir, 'processed')).mkdir(parents=True, exist_ok=True)
            try:
                os.remove(os.path.join(indir, 'processed', 'energies.pkl'))
            except FileNotFoundError:
                pass
            energies.to_pickle(os.path.join(indir, 'processed', 'energies.pkl'))
            logging.info("Calculated energies saved")

    if len(primaries) != len(energies):
        raise IndexError('Indices of primary information and MCTree do not align')

    df = pd.concat([primaries, energies], axis=1)
    df['event'] = np.arange(len(df))


    for e_label in ['energy', 'shower_energy', 'track_energy']:
        name = 'log10('+ e_label + ')'
        df[name] = np.log10(df[e_label] + eps)

    # Convert azimuth and zenith values to cartesian
    polar_directions = np.array(polar2cart(df['azimuth'].values, df['zenith'].values))
    polar_df = pd.DataFrame(polar_directions.T, columns=['x_dir', 'y_dir', 'z_dir'])

    # Get PID information (0 = cascade, 1 = track)
    df['PID'] = np.array(df['track_energy'] > 0).astype(int)

    # Get interaction type (1 = NC, 2 = CC); not available for Upgrade data yet
    try:
        weight_dict = np.load(os.path.join(indir, 'I3MCWeightDict', 'data.npy'))
        interaction_df = pd.DataFrame(weight_dict['InteractionType'], columns=['interaction_type'])
        weight = pd.DataFrame(weight_dict['weight'], columns=['weight'])
        df = pd.concat([df, interaction_df, weight], axis=1)
    except FileNotFoundError:
        pass

    return pd.concat([df, polar_df], axis=1)

#
# def join_pulse_info(gcd, indirs, pulse_frame, file_index, event_index, tempdir):
#         pulse_info = np.memmap(os.path.join(tempdir, 'pulse_info.npy'), mode='w+', shape=(file_index[-1, 1], 5), dtype=np.float)
#         # pulse_info = np.memmap(os.path.join(tempdir, 'pulse_info.npy'), mode='w+', shape=(np.sum(npulses_per_indir), 7), dtype=np.float)
#         pulse_index = np.memmap(os.path.join(tempdir, 'pulse_index.npy'), mode='w+', shape=(event_index[-1, 1], 2), dtype=np.int)
#         for indir, (f_start, f_stop), (e_start, e_stop) in zip(indirs, file_index, event_index):
#             data = open_memmap(os.path.join(indir, pulse_frame, 'data.npy'))
#             index = open_memmap(os.path.join(indir, pulse_frame, 'index.npy'))
#             pulse_info[f_start:f_stop, 0:3] = gcd[data['key']['string']-1, data['key']['om']-1]
#             pulse_info[f_start:f_stop, 3] = data['pulse']['time']
#             pulse_info[f_start:f_stop, 4] = data['pulse']['charge']
# #             pulse_info[f_start:f_stop, 5] = data['pulse']['flags'] & 1
# #             pulse_info[f_start:f_stop, 6] = (data['pulse']['flags'] & 2)/2
#
#             pulse_index[e_start:e_stop, 0] = index['start'] + f_start
#             pulse_index[e_start:e_stop, 1] = index['stop'] + f_start
#         return pulse_info, pulse_index
#
# def join_pulse_info_upgrade(gcd, indirs, pulse_frame, file_index, event_index, tempdir):
#         pulse_info = np.memmap(os.path.join(tempdir, 'pulse_info.npy'), mode='w+', shape=(file_index[-1, 1], 12), dtype=np.float)
#         pulse_index = np.memmap(os.path.join(tempdir, 'pulse_index.npy'), mode='w+', shape=(event_index[-1, 1], 2), dtype=np.int)
#         for indir, (f_start, f_stop), (e_start, e_stop) in zip(indirs, file_index, event_index):
#             data = open_memmap(os.path.join(indir, pulse_frame, 'data.npy'))
#             index = open_memmap(os.path.join(indir, pulse_frame, 'index.npy'))
#             string = data['key']['string'] - 1
#             om = data['key']['om'] - 1
#             pmt = data['key']['pmt']
#             xyz = gcd['geo'][string, om, pmt]
#             pulse_info[f_start:f_stop, 0:3] = xyz
#             pulse_info[f_start:f_stop, 3] = data['pulse']['time']
#             pulse_info[f_start:f_stop, 4] = data['pulse']['charge']
#
#             direction = gcd['direction'][string, om, pmt]
#             cart_dirs = np.array(polar2cart(*direction.T)).T
#             pulse_info[f_start:f_stop, 5:8] = cart_dirs
#
#             omdict = {'IceCube': 0, 'PDOM':1, 'mDOM':2, 'DEgg':3}
#             om_codes = np.array([omdict[gcd['omtype'][s, o, m].decode('UTF-8')]
#                         for s, o, m in zip(string, om, pmt)])
#             pulse_info[f_start:f_stop, 8:12] = 0
#             # one-hot encode
#             for i in omdict.values():
#                 idx = np.where(om_codes==i)[0]
#                 pulse_info[f_start:f_stop, 8+i][idx] = 1
# #             pulse_info[f_start:f_stop, -2] = data['pulse']['flags'] & 1
# #             pulse_info[f_start:f_stop, -1] = (data['pulse']['flags'] & 2)/2
#
#             pulse_index[e_start:e_stop, 0] = index['start'] + f_start
#             pulse_index[e_start:e_stop, 1] = index['stop'] + f_start
#         return pulse_info, pulse_index

def convert_into_memmap(memmap, data, gcd):
    memmap[:, 0:3] = gcd[data['key']['string'] - 1, data['key']['om'] - 1]
    memmap[:, 3] = data['pulse']['time']
    memmap[:, 4] = data['pulse']['charge']
    # memmap[:, 5] = data['pulse']['flags'] & 1
    # memmap[:, 6] = (data['pulse']['flags'] & 2)/2
    pass

def convert_into_memmap_upgrade(memmap, data, gcd):
    string = data['key']['string'] - 1
    om = data['key']['om'] - 1
    pmt = data['key']['pmt']
    xyz = gcd['geo'][string, om, pmt]
    memmap[:, 0:3] = xyz
    memmap[:, 3] = data['pulse']['time']
    memmap[:, 4] = data['pulse']['charge']

    direction = gcd['direction'][string, om, pmt]
    cart_dirs = np.array(polar2cart(*direction.T)).T
    memmap[:, 5:8] = cart_dirs

    omdict = {'IceCube': 0, 'PDOM': 1, 'mDOM': 2, 'DEgg': 3}
    om_codes = np.array([omdict[gcd['omtype'][s, o, m].decode('UTF-8')]
                         for s, o, m in zip(string, om, pmt)])
    memmap[:, 8:12] = 0
    # one-hot encode
    for i in omdict.values():
        idx = np.where(om_codes == i)[0]
        memmap[:, 8 + i][idx] = 1

    pass
