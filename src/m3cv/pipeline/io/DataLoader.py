import os
import h5py
import json
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from m3cv.pipeline.io.data_augmentation import zoom, rotate, shift, downsample
from m3cv.pipeline.io._utils import (
    window_level, get_unique_values, rebuild_mask, get_survival,
    stats_continuous, stats_categorical
)
from m3cv.pipeline.io._datautils import (
    is_valid,
    _label_logic_from_config
)
from m3cv._coreutils import simplelog

class DataManager:

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
        self.encoders
        for k,v in self.active_fields.items():



class Preprocessor:

    def __init__(self,data):
        try:
            fit_to = np.array(data, dtype=np.float32)
            fit_to = fit_to[fit_to!=np.inf].reshape(-1,1)
        except ValueError:
            fit_to = np.array(data).reshape(-1,1)
        if fit_to.dtype == np.float32:
            self._operator = MinMaxScaler()
        else:
            self._operator = OneHotEncoder(
                sparse=True, handle_unknown='ignore'
            )
        self._operator.fit(fit_to)
    
    def __call__(self,data):
        return self._operator.transform(data)
