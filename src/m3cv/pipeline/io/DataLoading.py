import os
import h5py
import json
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from m3cv.pipeline.preprocessing.data_augmentation import zoom, rotate, shift, downsample

from m3cv.pipeline.io._datautils import (
    is_valid,
    _label_logic_from_config,
    rebuild_sparse,
    split_list
)
from m3cv._coreutils import simplelog

class DataLoader:

    def __init__(self, config):
        self.config = config
        self.root = self.config.data.processed_data_path
        self.files = [
            file for file in os.listdir(self.root) if file.endswith('.h5')
            ]
        self.scouted = False
        supp = self.config.data.extraction.supplemental_data.scan()
        self.primary = self.config.data.preprocessing.dynamic.modalities
        self.supp = {k:pd.DataFrame() for k in supp}
        self.active_fields = {
            k:getattr(config.data.preprocessing.dynamic.supplemental,k) \
                for k in self.supp.keys()
        }

    def scout_files(self, verbose=False):
        """Traverses file list to check for validity as well as load tabular 
        data into memory for quick access.
        """
        if self.scouted:
            return
        
        simplelog("Scouting - validity check, collect supp. data...", verbose)
        one_tenth = len(self.files) // 10
        for i,file in enumerate(self.files[:]):
            with h5py.File(os.path.join(self.root, file),'r') as f:
                # check validity
                if not is_valid(f, self.primary, list(self.supp.keys())):
                    print(f"{file} missing required contents, skipping...")
                    self.files.remove(file)
                    continue

                for s in self.supp.keys():
                    temp = pd.DataFrame(
                        columns=f[s].attrs['fields'].astype(str),
                        index=[file],
                        data=f[s][...].astype(str).reshape(1,-1)
                        )
                    self.supp[s] = pd.concat(
                        [self.supp[s], temp]
                    )
            if (i+1) % one_tenth == 0:
                simplelog(f"Done with {i+1}/{len(self.files)}...", verbose)
        # clean out any all-nan columns
        for df in self.supp.values():
            for field in df.columns[:]:
                if (df[field].astype(str)=='nan').all():
                    df.drop(columns=[field], inplace=True)
        simplelog("Scout of files is complete.", verbose)
        self.scouted = True

    def make_binary_labels(self, override=None):
        """Calculates endpoint labels based on provided logic.
        Defaults to pulling from config, but you can manually override.
        If override, must pass a dictionary defining both positive and
        negative. Logic format must be list of strings that walk the
        function to the right location in supplemental data, ex:
            {'positive':['clinical', 'Date of Death', '< 730']}

        TODO - build override capability
        """
        self.scout_files()
        self.labels = _label_logic_from_config(
            self.config, self.files, self.supp
        )

    def build_encoders(self):
        self.scout_files()
        self.encoders = {}
        for k in self.active_fields.keys():
            self.encoders[k] = {}
        for k,values in self.active_fields.items():
            for v in values:
                self.encoders[k][v] = encoder(
                    self.supp[k][v]
                )

    @property
    def supp_vector_len(self):
        # length of the 1D array output of nonvolume data for each patient
        length = 0
        for k,values in self.active_fields.items():
            for v in values:
                if isinstance(self.encoders[k][v],OneHotEncoder):
                    length += len(self.encoders[k][v].categories_[0])
                elif isinstance(self.encoders[k][v],MinMaxScaler):
                    length += 1
        return length
    
    @property
    def supp_vector_desc(self):
        desc = []
        for k,values in self.active_fields.items():
            for v in values:
                if isinstance(self.encoders[k][v],OneHotEncoder):
                    to_add = list(self.encoders[k][v].categories_[0])
                    to_add = ['{}-{}: {}'.format(k,v,x) for x in to_add]
                    desc += to_add
                elif isinstance(self.encoders[k][v],MinMaxScaler):
                    desc.append(f'{k} - {v}')
        return desc
    
    @property
    def channels_map(self):
        channels = []
        for channel in self.config.data.preprocessing.dynamic.modalities:
            if isinstance(channel, dict):
                for mask in channel['ROI']:
                    channels.append(f'ROI - {mask}')
            else:
                channels.append(channel)
        return channels

    def load_patient(self, file):
        with h5py.File(os.path.join(self.root, file), 'r') as f:
            # load volumes
            volumes = []
            refshape = f.attrs['shape']
            for channel in self.config.data.preprocessing.dynamic.modalities:
                if isinstance(channel, dict):
                    for mask in channel['ROI']:
                        volumes.append(rebuild_sparse(
                            f[channel][mask]['slices'],
                            f[channel][mask]['rows'],
                            f[channel][mask]['cols'],
                            refshape
                        ))
                if isinstance(f[channel], h5py.Dataset):
                    volumes.append(f[channel][...])
                elif isinstance(f[channel], h5py.Group):
                    volumes.append()
            volume = np.stack(volumes, axis=-1)

            nonvolume = []
            for k,values in self.active_fields.items():
                for v in values:
                    fieldmap = f[k].attrs['fields'].astype(str)
                    entry = f[k][np.where(fieldmap==v)].astype(str)
                    entries = entry[0].split("|") # hardcoded delimiter
                    entries = [e if e != 'inf' else 'nan' for e in entries]
                    conversion = True if isinstance(
                        self.encoders[k][v], MinMaxScaler
                        ) else False
                    if conversion:
                        nonvolume += [
                            self.encoders[k][v].transform(
                                np.array(e, dtype=np.float32).reshape(-1,1)
                            ) for e in entries
                        ]
                    else:
                        temp = np.zeros(
                            shape=(1,len(self.encoders[k][v].categories_[0]))
                            )
                        for e in entries:
                            temp += self.encoders[k][v].transform(
                                np.array(e).reshape(-1,1)
                            )
                        nonvolume += [temp]
            if len(nonvolume) <= 1:
                nonvolume = np.array(nonvolume)
            else:
                nonvolume = np.concatenate(nonvolume, axis=1)
            nonvolume = np.squeeze(nonvolume)
            nonvolume[np.isnan(nonvolume)] = 0

        return volume, nonvolume, self.labels.at[file]

class Handler:
    def __init__(self, loader: DataLoader, batch_size=16):
        self.loader = loader
        self.batch_size = batch_size
        if not hasattr(loader, 'labels'):
            loader.make_binary_labels()
        self.pos = loader.labels[loader.labels == 1].index.to_list()
        self.neg = loader.labels[loader.labels == 0].index.to_list()
        self.invalid = loader.labels[loader.labels == 99].index.to_list()
        self.valid = self.pos + self.neg
        self.all = self.pos + self.neg + self.invalid
        self.preloaded = False

    

    def set_split(
            self,
            frac_map={'train':0.9,'test':0.1},
            stratified=True,
            seed=None
            ):
        if seed:
            np.random.seed(seed)
        if not stratified:
            np.random.shuffle(self.valid)
            splits = split_list(self.valid, list(frac_map.values()))
            for k,l in zip(frac_map.keys(), splits):
                setattr(self,k,l)
        elif stratified:
            np.random.shuffle(self.pos)
            np.random.shuffle(self.neg)
            pos_splits = split_list(self.pos, list(frac_map.values()))
            neg_splits = split_list(self.neg, list(frac_map.values()))
            merged_splits = [p + n for p,n in zip(pos_splits, neg_splits)]
            for k,l in zip(frac_map.keys(), merged_splits):
                setattr(self,k,l)

    def kfolds(self, seed, nfolds=10, stratified=True, testsplit=0):
        # function to establish kfolds
        np.random.seed(seed)
        if not stratified:
            np.random.shuffle(self.valid)
            self.splits = np.array_split(self.valid, nfolds)
        else:
            np.random.shuffle(self.pos)
            np.random.shuffle(self.neg)
            pos_splits = np.array_split(self.pos, nfolds)
            neg_splits = np.array_split(self.neg, nfolds)
            self.splits = [
                np.concatenate([p,n],axis=0) 
                for p,n in zip(pos_splits, neg_splits)
                ]
        self.set_test_split(testsplit)
            
    def set_test_split(self, testsplit):
        # used to change which split is the test split. takes int arg
        self.test = self.splits[testsplit]
        self.train = [
            self.splits[i] for i in range(len(self.splits)) if i != testsplit
            ]
        self.train = np.concatenate(self.train, axis=0)
            
    @property
    def train_ceiling(self):
        """
        The tf.Dataset from generator structure doesn't play nice with leftover
        partial batches at the end of epochs, so this value is used to get a
        clean ceiling that ensures complete batches. Obviously this leaves
        some remainder of patients out of any given epoch, however, since the
        generator shuffles the list at each reset, over multiple epochs every
        data element is expected to be included at some point.

        Cannot be called until splits are defined.
        """
        if self.batch_size is None:
            c = len(self.train)
        else:
            c = len(self.train) - (len(self.train) % self.batch_size)
        if self.flip_double is True:
            c *= 2
        return c
            
    def preload(self):
        # attempts to preload data, assuming there's sufficient memory space
        for x in ['trXv', 'trXnv', 'trY', 'teXv', 'teXnv', 'teY']:
            setattr(self, x, [])
        for pt in self.train:
            v, nv, y = self.loader.load_patient(pt)
            self.trXv.append(v)
            self.trXnv.append(nv)
            self.trY.append(y)
        self.trXv = np.stack(self.trXv, axis=0)
        self.trXnv = np.stack(self.trXnv, axis=0)
        self.trY = np.stack(self.trY, axis=0)
        self.train_map = {
            pt:i for pt, i in zip(
                self.train,np.arange(len(self.train))
                )
            }

        for pt in self.test:
            v, nv, y = self.loader.load_patient(pt)
            self.teXv.append(v)
            self.teXnv.append(nv)
            self.teY.append(y)
        self.teXv = np.stack(self.teXv, axis=0)
        self.teXnv = np.stack(self.teXnv, axis=0)
        self.teY = np.stack(self.teY, axis=0)
        self.test_map = {
            pt:i for pt, i in zip(
                self.test,np.arange(len(self.test))
                )
            }

        self.preloaded = True

    def __call__(self):
        """call_reference is a list of indices that map to the fixed-order
        of the train patient list and, if applicable, preloaded data. This 
        way we can simply shuffle the reference, without needing to perform
        synchronized shuffling of multiple other arrays.
        """
        self.call_index = 0
        self.call_reference = np.arange(len(self.train))
        np.random.shuffle(self.call_reference)
        while True:
            if self.call_index >= self.train_ceiling:
                self.call_index = 0
                np.random.shuffle(self.call_reference)
            pt = self.train[self.call_reference[self.call_index]]

            if self.preloaded:
                v = self.trXv[self.train_map[pt],...]
                nv = self.trXnv[self.train_map[pt],...]
                y = self.trY[self.train_map[pt],...]
            else:
                v, nv, v = self.loader(pt)

            if hasattr(self, 'augmenter'):
                v = self.augmenter(v) #TODO - build augmenter
            yield (v, nv), y
            self.call_index += 1

def encoder(data):
    try:
        fit_to = np.array(data, dtype=np.float32)
        fit_to = fit_to[fit_to!=np.inf].reshape(-1,1)
    except ValueError:
        fit_to = np.array(data).reshape(-1,1)
    if fit_to.dtype == np.float32:
        operator = MinMaxScaler()
    else:
        operator = OneHotEncoder(
            sparse=True, handle_unknown='ignore'
        )
    operator.fit(fit_to)
    return operator
