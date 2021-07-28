import torch
import logging
from torch_geometric.data import Data
from torch_geometric.data import Dataset as TorchSet
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import h5py
import pickle
from numpy.lib.format import open_memmap
from importlib.resources import path
import os
from joblib import Parallel, delayed
from nodalinterchange.utils.i3cols_to_pandas import get_truths, convert_into_memmap, convert_into_memmap_upgrade
from nodalinterchange.utils.dataloader import get_pulses as get_pulses_hf5
from nodalinterchange.utils.icecubeutils import get_dom_positions
from nodalinterchange.utils.datautils import flatten, RangeDict
import nodalinterchange.resources


class Dataset(TorchSet):
    """
    Class to load, transform, and hold data for training and evaluation.
    Parameters:
    ----------
    indir_list: list
        List of paths pointing to i3cols output
    upgrade: bool
        Process input data for IceCube Upgrade
    truth_labels: list
        List of truth frame names to put into data objects
    save_energies: bool
        Save calculated energies to input directory
    force_recalculate: bool
        Recalculate energies
    make_data_list: bool
        Make list of Pytorch Data objects after loading and processing data
    """
    # TODO: maybe add option to select which pulse frame to use (SRT TW cleaned pulses most usual though)
    # TODO: add option to choose which features and truths to use
    # TODO: maybe add option to move normalization and datalist creation to trainer
    # TODO: add filter
    # TODO: function to combine Datasets
    # TODO: add functions to set features / truths
    # TODO: Check if resetting indices of dataframes is needed in converter
    def __init__(self,
                 indir_list,
                 upgrade=False,
                 # feature_labels=None,
                 truth_labels=['x', 'y', 'z', 'x_dir', 'y_dir', 'z_dir', 'log10(energy)', 'log10(shower_energy)', 'log10(track_energy)', 'PID'],
                 save_energies=False,
                 force_recalculate=False,
                 overwrite_data=True,
                 normalization_parameters=None,
                 make_data_list=True):
        super().__init__()
        self.files = indir_list
        self.upgrade = upgrade
        logging.info('Loading inputs')
        self._truths, self._file_lengths = self._load_inputs(save_energies, force_recalculate)
        self._pulse_frame = "SRTTWOfflinePulsesDC" if not self.upgrade else "SplitInIcePulsesSRT"
        self._pulse_index, self._event_index = self._get_input_information(self._pulse_frame)
        self._range_dict = RangeDict()
        for i, (start, stop) in enumerate(self._event_index):
            self._range_dict[start:stop] = i
        self._truths.reset_index(drop=True, inplace=True)
        self.write_data(overwrite_data)
        self.non_empty_mask = self._get_non_empty_events()
        self.filter = np.arange(len(self.non_empty_mask))
        if normalization_parameters is not None:
            self._means = normalization_parameters['means']
            self._stds = normalization_parameters['stds']
        else:
            self._means, self._stds = None, None
            self._read_normalization_parameters()
            pass
        self.truth_labels = truth_labels
        logging.info("Dataset setup finished")

    def _load_inputs(self, save_energies=False, force_recalculate=False):
        """
        Load events and truths from input directories
        Increment event index based on files loaded
        """
        if not isinstance(self.files, list):
            raise TypeError('Input directories have to be list')
        elif len(self.files) == 0:
            raise ValueError('Input list empty')
        elif len(self.files) == 1:
            indir = self.files[0]
            truths = get_truths(indir,
                                save_energies=save_energies,
                                force_recalculate=force_recalculate)
            return truths, [len(truths)]
        else:
            truths = [get_truths(indir,
                                 save_energies=save_energies,
                                 force_recalculate=force_recalculate)
                      for indir in self.files]
            # Increment event indices
            last_event_idx = np.array([
                np.max(np.unique(frame['event'].values))
                for frame in truths
            ])
            self._file_stops = last_event_idx
            increments = np.cumsum(np.concatenate([[0], (np.array(last_event_idx[:-1])+1)]))
            for truth_frame, inc in zip(truths, increments):
                truth_frame['event'] += inc
            lengths = [len(t) for t in truths]
            truths_all = pd.concat(truths)
            return truths_all, lengths

    def _get_input_information(self, pulse_frame):
        indirs = self.files
        npulses_per_indir = []
        nevents_per_indir = []
        for indir in indirs:
            data = open_memmap(os.path.join(indir, pulse_frame, 'data.npy'))
            npulses_per_indir.append(len(data))
            index = open_memmap(os.path.join(indir, pulse_frame, 'index.npy'))
            nevents_per_indir.append(len(index))

        get_index_from_lengths = lambda lengths: np.vstack([
            np.concatenate([[0], np.cumsum(lengths)[:-1]]),
            np.cumsum(lengths),
            ]).T
        pulse_index = get_index_from_lengths(npulses_per_indir)
        event_index = get_index_from_lengths(nevents_per_indir)
#         if np.sum(nevents_per_indir) != len(self._truths):
#             print(np.sum(nevents_per_indir), len(self._truths))
#             raise IndexError('Pulses and truths do not line up')
        return pulse_index, event_index

    def write_data(self, overwrite_data):
        for indir in self.files:
            processed_dir = os.path.join(indir, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            data = open_memmap(os.path.join(indir, self._pulse_frame, 'data.npy'))

            if not self.upgrade:
                with path(nodalinterchange.resources, 'geo_array.npy') as p:
                    gcd_path = p
                gcd = open_memmap(gcd_path)
            else:
                with path(nodalinterchange.resources, 'icu_gcd.p') as p:
                    gcd_path = p
                gcd = pickle.load(open(gcd_path, 'rb'), encoding='latin1')

            if overwrite_data:
                try:
                    os.remove(os.path.join(processed_dir, 'data.npy'))
                except FileNotFoundError:
                    pass

            n_features = 5 if not self.upgrade else 12
            data_converted = open_memmap(os.path.join(processed_dir, 'data.npy'),
                                         dtype=np.float,
                                         shape=(len(data), n_features),
                                         mode='w+'
                                         )
            if not self.upgrade:
                convert_into_memmap(data_converted, data, gcd)
            else:
                convert_into_memmap_upgrade(data_converted, data, gcd)

            if overwrite_data:
                try:
                    os.remove(os.path.join(processed_dir, 'event_info.h5'))
                except FileNotFoundError:
                    pass
            with h5py.File(os.path.join(processed_dir, 'event_info.h5'), 'w') as file_info:
                file_info['n_events'] = len(data)
                feature_sum = np.array([np.sum(f) for f in data_converted.T])
                file_info['sum'] = feature_sum
                feature_squared_sum = np.array([np.sum(f**2) for f in data_converted.T])
                file_info['squared_sum'] = feature_squared_sum
        pass

    def add_truth(self, df, overwrite=False):
        intersect = np.intersect1d(df.columns, self._truths.columns)
        df.reset_index(drop=True, inplace=True)
        if len(intersect) != 0 and not overwrite:
            raise IndexError('Names already exist:', intersect, '; Choose different column names')
        elif overwrite:
            for col in intersect:
                del self._truths[col]
        self._truths = pd.concat([self._truths, df], axis=1)

    def _read_normalization_parameters(self):
        n_events = 0
        sums = 0
        squared_sums = 0
        for indir in self.files:
            with h5py.File(os.path.join(indir, 'processed', 'event_info.h5'), 'r') as f:
                n_events += np.array(f['n_events'])
                sums += np.array(f['sum'])
                squared_sums += np.array(f['squared_sum'])
        means = sums / n_events
        variances = squared_sums / n_events - means**2
        self._means = torch.tensor(means, dtype=torch.float)
        self._stds = torch.tensor(np.sqrt(variances), dtype=torch.float)
        pass

    def _get_non_empty_events(self):
        indices = []
        n_events = []
        for indir in self.files:
            index = open_memmap(os.path.join(indir, self._pulse_frame, 'index.npy'))
            indices.append(np.where(index['start'] != index['stop'])[0])
            n_events.append(len(index))
        mask = np.array(indices[0])
        if len(indices) > 1:
            for i, idx in enumerate(indices[1:]):
                mask = np.append(mask, idx + n_events[i+1])
        return mask

    def set_normalization_parameters(self, norm_pars):
        means = norm_pars['means']
        stds = norm_pars['stds']
        self._means = torch.tensor(means, dtype=torch.float)
        self._stds = torch.tensor(stds, dtype=torch.float)
        logging.info("New normalization parameters set")
        pass

    def set_filter(self, filter_):
        if self.non_empty_mask is None:
            logging.warning('Setting filter before initial data list building not recommended')
        self.filter = filter_
        logging.info('Filter set')

    def reset_filter(self):
        self.filter = np.arange(len(self.non_empty_mask))
        logging.info('Filter removed')

    def len(self):
        if self.filter is not None:
            return len(self.filter)
        return len(self.non_empty_mask)

    def get_as_numpy(self, idx):
        i = self.non_empty_mask[self.filter[idx]]
        indir_idx, increment = self._range_dict[i]
        index = open_memmap(os.path.join(self.files[indir_idx], self._pulse_frame, 'index.npy'))
        data = open_memmap(os.path.join(self.files[indir_idx], 'processed', 'data.npy'))
        start, stop = index[i]
        # truth = self._truths.iloc[i][self.truth_labels]
        d = Data(x=torch.tensor(data[start:stop], dtype=torch.float),
                 y=torch.tensor(truth, dtype=torch.float))
        d.x = torch.div(torch.sub(d.x, self._means), self._stds) # normalize
        d = (data[start:stop] - self._means) / self._stds
        return d

    def get(self, idx):
        i = self.non_empty_mask[self.filter[idx]]
        indir_idx, increment = self._range_dict[i]
        index = open_memmap(os.path.join(self.files[indir_idx], self._pulse_frame, 'index.npy'))
        data = open_memmap(os.path.join(self.files[indir_idx], 'processed', 'data.npy'))
        start, stop = index[i]
        truth = self._truths.iloc[i][self.truth_labels]
        d = Data(x=torch.tensor(data[start:stop], dtype=torch.float),
                 y=torch.tensor(truth, dtype=torch.float))
        d.x = torch.div(torch.sub(d.x, self._means), self._stds) # normalize
        return d

    def get_truths(self):
        idx = self.non_empty_mask[self.filter]
        return self._truths.iloc[idx]

    def set_truth_labels(self, labels):
        self.truth_labels = labels


class DatasetFromData(Dataset):
    # TODO: maybe add option to select which pulse frame to use (SRT TW cleaned pulses most usual though)
    # TODO: add option to choose which features and truths to use
    # TODO: maybe add option to move normalization and datalist creation to trainer
    # TODO: add filter
    # TODO: function to combine Datasets
    # TODO: add functions to set features / truths
    # TODO: Check if resetting indices of dataframes is needed in converter
    def __init__(self,
            indir_list,
            upgrade=False,
            feature_labels=None,
            truth_labels=['x', 'y', 'z', 'x_dir', 'y_dir', 'z_dir', 'log10(energy)', 'log10(shower_energy)', 'log10(track_energy)', 'PID']):
        self.files = indir_list
        self.upgrade = upgrade
        self.truth_labels = truth_labels
        if feature_labels is None and not self.upgrade:
            self.feature_labels = ['x_om', 'y_om', 'z_om', 'time', 'charge']
        elif feature_labels is None and self.upgrade:
            self.feature_labels = ['x_om', 'y_om', 'z_om', 'time', 'charge', 'xdir_om', 'ydir_om', 'zdir_om', 'is_IceCube', 'is_PDOM', 'is_mDOM', 'is_DEgg']
        self.raw_pulses = self._load_inputs()
        # self._non_empty_mask = self.truths['event'].isin(self.raw_pulses['event'].unique()) # Remove truths of empty pulses
        # self.truths = self.truths[self._non_empty_mask]
        self.raw_pulses.reset_index(drop=True)
        # self.truths.reset_index(drop=True, inplace=True)
        self._means, self._stds = self._get_normalization_parameters()
        self.normalized_pulses = None
        self._make_normalized_pulses(self._means, self._stds)
        self.normalization_parameters = {'means':  self._means, 'stds': self._stds}
        self.data_list = None
        self.n_events = None
        self.make_data_list()

    def _load_inputs(self):
        """
        Load events and truths from input directories
        Increment event index based on files loaded
        """
        if not isinstance(self.files, list):
            raise TypeError('Input directories have to be list')
        elif len(self.files) == 0:
            raise ValueError('Input list empty')
        elif len(self.files) == 1:
            indir = self.files[0]
            return get_pulses(indir, self.upgrade), get_truths(indir, save_energies=save_energies)
        # TODO: Make index checks
        else:
            events = [get_pulses(indir, self.upgrade) for indir in self.files]
            # Increment event indices
            last_event_idx = np.array([np.max(np.unique(frame['event'])) for frame in events])
            increments = np.cumsum(np.concatenate([[0], (np.array(last_event_idx[:-1])+1)]))
            for event_frame, inc in zip(events, increments):
                event_frame['event'] += inc
            events = dd.concat(events, ignore_index=True, sort=False)
            return events

    def make_data_list(self, truth_labels=None):
        event_list = dataframe_to_event_list(self.normalized_pulses)
        data_list = [Data(x=torch.tensor(x, dtype=torch.float)) for x in event_list]
        self.data_list = data_list
        self.n_events = len(data_list)


class DatasetFromHDF5(Dataset):
    def __init__(self,
            indir_list,
            upgrade=False,
            feature_labels=None,
            n_jobs=1,
            make_data_list=False):
        self.n_jobs = n_jobs
        super().__init__(indir_list, upgrade=upgrade, feature_labels=feature_labels, make_data_list=make_data_list)

    def _load_inputs(self, *args, **kwargs):
        logging.warning('PID INFORMATION USELESS, DO NOT USE! \n If PID is necessary, use standard Dataset')
        labels = ['x', 'y', 'z', 'time', 'azimuth', 'zenith', 'energy']
        # file_inputs = [get_pulses(f, labels=labels) for f in tqdm(self.files)]
        file_inputs = Parallel(n_jobs=self.n_jobs)(delayed(get_pulses_hf5)(f, labels=labels) for f in tqdm(self.files))
        pulses, raw_truths = map(list, zip(*file_inputs))
        pulses = flatten(pulses)
        raw_truths = flatten(raw_truths)

        dom_positions = get_dom_positions()
        hit_doms = [dom_positions[event[:,0].astype(int)] for event in pulses]
        raw_pulses = [np.concatenate((
            doms, # xyz
            event[:,1].reshape(-1,1), # Time
            event[:,2].reshape(-1,1), # charge
                                   ), axis=1)
                    for event, doms in zip(pulses, hit_doms)]
        n_pulses = [len(event) for event in raw_pulses]
        event_idx = np.concatenate([np.full(l, i) for i, l in enumerate(n_pulses)])
        raw_pulses = np.concatenate(raw_pulses)
        raw_pulses = np.concatenate([raw_pulses, event_idx.reshape(-1, 1)], axis=1)
        raw_pulses_df = pd.DataFrame(raw_pulses, columns=['x_om', 'y_om', 'z_om', 'time', 'charge', 'event'])

        # Fake PID
        n_events = len(raw_truths)
        fake_pid = np.zeros(n_events).reshape(-1, 1)
        raw_truths = np.concatenate((np.array(raw_truths), fake_pid), axis=1)
        labels.append('track_energy')
        labels[labels.index('energy')] = 'neutrino_energy'
        truths_df = pd.DataFrame(raw_truths, columns=labels)
        event_df = pd.DataFrame(np.arange(len(raw_truths)), columns=['event'])
        truths_df = pd.concat([truths_df, event_df], axis=1)
        input_dict = {label: idx for idx, label in enumerate(labels)}
        return raw_pulses_df, truths_df
        # return raw_pulses, raw_truths, input_dict