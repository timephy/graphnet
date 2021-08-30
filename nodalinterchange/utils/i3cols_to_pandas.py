import numpy as np
import logging
from numpy.lib.format import open_memmap
from numpy.lib.recfunctions import merge_arrays, unstructured_to_structured
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
    neutrino_energies = np.array(energies[tree_idx['start']], dtype=[('neutrino_energy', float)])
    nevents = len(tree_idx)
    additional_info = np.zeros(dtype=[('track_energy', float), ('track_length', float), ('shower_energy', float)],
                                   shape=nevents)

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

    if len(muon_idx) != 0:
        additional_info['track_energy'][has_track] = energies[muon_idx]
        additional_info['track_length'][has_track] = tree_data['particle']['length'][muon_idx]
        additional_info['shower_energy'] = neutrino_energies['neutrino_energy'] - (additional_info['track_energy'] + e_invis)
    df = merge_arrays([neutrino_energies, additional_info], flatten=True)
    return df

def load_primary_information(mcprimary):
    # Have to split this in two parts because of hierarchical structure of input data
    # maybe there's a smarter way of doing this
    labels = ['pdg_encoding', 'time', 'energy']
    superlabels = ['pos', 'dir']
    df_0 = np.copy(mcprimary[labels])
    df_1 = merge_arrays([np.copy(mcprimary[l]) for l in superlabels], flatten=True)

    return merge_arrays([df_0, df_1], flatten=True)

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
        energies = np.load(os.path.join(indir, 'processed', 'energies.npy'))
        logging.info('Loading precalculated cascade and track energy information')
        if save_energies:
            logging.warning('Processed cascade and track energy information found, not saving to file')
    except FileNotFoundError:
        energies = get_energies(mctree_data, mctree_idx)
        if save_energies:
            logging.info('Saving cascade and track information')
            Path(os.path.join(indir, 'processed')).mkdir(parents=True, exist_ok=True)
            try:
                os.remove(os.path.join(indir, 'processed', 'energies.npy'))
            except FileNotFoundError:
                pass
            np.save(os.path.join(indir, 'processed', 'energies.npy'), energies)
            logging.info("Calculated energies saved")

    if len(primaries) != len(energies):
        print(len(primaries), len(energies))
        raise IndexError('Indices of primary information and MCTree do not align')

    df = merge_arrays([primaries, energies], flatten=True)

    for e_label in ['neutrino_energy', 'shower_energy', 'track_energy']:
        name = 'log10('+ e_label + ')'
        arr = np.array(np.log10(df[e_label] + eps), dtype=[(name, float)])
        df = merge_arrays([df, arr], flatten=True)

    # Convert azimuth and zenith values to cartesian
    polar_directions = np.array(polar2cart(df['azimuth'], df['zenith']))
    polar_df = unstructured_to_structured(polar_directions.T, names=['x_dir', 'y_dir', 'z_dir'])

    # Get PID information (0 = cascade, 1 = track)
    df = merge_arrays([df, np.array(df['track_energy'] > 0, dtype=[('PID', float)])], flatten=True)

    # Get interaction type (1 = NC, 2 = CC); not available for Upgrade data yet
    try:
        weight_dict = np.load(os.path.join(indir, 'I3MCWeightDict', 'data.npy'))
        df = merge_arrays([df, weight_dict['InteractionType'], weight_dict['weight']], flatten=True)
    except FileNotFoundError:
        pass

    return merge_arrays([df, polar_df], flatten=True)


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

# def convert_into_memmap_upgrade(memmap_loc, data, gcd):
#     keys = [
#         'x',
#         'y',
#         'z',
#         'time',
#         'charge',
#         'pmt_xdir',
#         'pmt_ydir',
#         'pmt_zdir',
#         'is_IceCube',
#         'is_PDOM',
#         'is_mDOM',
#         'is_DEgg',
#         'string',
#         'om',
#         'pmt',
#         'is_LC',
#         'has_ATWD'
#     ]
#     # assign float to every key because PyTorch needs floats later anyway
#     dtype = [(key, float) for key in keys]
#
#     memmap = open_memmap(memmap_loc,
#                          dtype=dtype,
#                          shape=len(data),
#                          mode='w+'
#                          )
#
#     string = data['key']['string'] - 1
#     om = data['key']['om'] - 1
#     pmt = data['key']['pmt']
#
#     memmap['string'] = string
#     memmap['om'] = om
#     memmap['pmt'] = pmt
#
#     xyz = gcd['geo'][string, om, pmt]
#     memmap['x'] =  xyz[:, 0]
#     memmap['y'] =  xyz[:, 1]
#     memmap['z'] =  xyz[:, 2]
#     memmap['time'] = data['pulse']['time']
#     memmap['charge'] = data['pulse']['charge']
#
#     direction = gcd['direction'][string, om, pmt]
#     cart_dirs = np.array(polar2cart(*direction.T)).T
#     memmap['pmt_xdir'] = cart_dirs[:, 0]
#     memmap['pmt_ydir'] = cart_dirs[:, 1]
#     memmap['pmt_zdir'] = cart_dirs[:, 2]
#
#     omdict = {'IceCube': 0, 'PDOM': 1, 'mDOM': 2, 'DEgg': 3}
#     om_codes = np.array([omdict[gcd['omtype'][s, o, m].decode('UTF-8')]
#                          for s, o, m in zip(string, om, pmt)])
#     # memmap[:, 8:12] = 0
#     # one-hot encode
#     for key, i in omdict.items():
#         idx = np.where(om_codes == i)[0]
#         memmap["is_" + key][idx] = 1
#
#     memmap['is_LC'] = data['pulse']['flags'] & 1
#     memmap['has_ATWD'] = (data['pulse']['flags'] & 2)/2
#
#     return memmap