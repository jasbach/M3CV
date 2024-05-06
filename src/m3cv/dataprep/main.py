from argparse import ArgumentParser
from m3cv.ConfigHandler import Config

from m3cv.dataprep.handler import Preprocessor
import m3cv.dataprep.arrayclasses as arrayclass
from m3cv.dataprep._preprocess_util import find_parotid_info, find_PTV_info

import os
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError



"""Script that houses the primary entrypoint for performing data packing.
"""

def run():
    parser = ArgumentParser(description='Preprocessor: DICOM to HDF5')
    parser.add_argument('configpath')
    args = parser.parse_args()
    with open(args.configpath, 'r') as f:
        config = Config(f)

# === here is the copy/pasted code to convert ===
    root = config.data.raw_data_path
        
    # Get the files - any file that errors on load will be skipped
    dcms = []
    patient_id = None
    for file in os.listdir(root):
        try:
            temp_dcm = pydicom.dcmread(os.path.join(root,file))
        except InvalidDicomError:
            continue # skip non DICOM files
        if patient_id is not None:
            patient_id = temp_dcm.PatientID
        else:
            assert temp_dcm.PatientID == patient_id, \
            "Root file must only contain one patient's DCM"
        dcms.append(temp_dcm)

    ct_files = []
    dose_files = []
    ss_files = []
    for dcm in dcms:
        if dcm.Modality == "CT":
            ct_files.append(dcm)
        elif dcm.Modality == "RTDOSE":
            dose_files.append(dcm)
        elif dcm.Modality == "RTSTRUCT":
            ss_files.append(dcm)
    
    if len(dose_files) == 1:
        dose_files = dose_files[0]
    if len(ss_files) > 1:
        raise Exception("Only one structure set file permitted in source dir.")
    ss = ss_files[0]
    
    ct_arr = arrayclass.PatientCT(ct_files)
    ct_arr.rescale(args.pixel_size)
    
    dose_arr = arrayclass.PatientDose(dose_files)
    dose_arr.align_with(ct_arr)
    
    # will need a refactor to genericize the ROI packing
    rois_to_build = []
    roi_proper_name = []
    for channel in config.data.extraction.modalities:
        if isinstance(channel, dict):
            for roi in channel['ROI']:
                if 'parotid' in roi.lower():
                    other_info = roi.lower().split('parotid')
                    if any(['r' in s for s in other_info]):
                        side = 'r'
                    elif any(['l' in s for s in other_info]):
                        side = 'l'
                    roi_name, roi_num = find_parotid_info(ss, side)
                    print("Looking for parotid - found:",roi_name, roi_num)
                    if roi_num is not None:
                        rois_to_build.append(roi_name)
                        roi_proper_name.append(f"parotid_{side}")

                elif 'ptv' in roi.lower():
                    roi_name, roi_num = find_PTV_info(ss)
                    print("Looking for PTV - found:",roi_name, roi_num)
                    if roi_num is not None:
                        rois_to_build.append(roi_name)
                        roi_proper_name.append('ptv')
    masks = []
    for roi_name, proper_name in zip(rois_to_build,roi_proper_name):
        temp = arrayclass.PatientMask(
            ct_arr, ss, roi_name, proper_name=proper_name
            )
        masks.append(temp)
            
    prepper = Preprocessor(patient_id=patient_id)
    prepper.attach([ct_arr, dose_arr])
    if len(masks) > 0:
        prepper.attach(masks)
    
    if 'eortc_qol' in config.data.supplemental_data.scan():
        surveyfile = pd.read_csv(
            config.data.supplemental_data.eortc_qol
        )
        prepper.populate_surveys(surveyfile)
        
    if 'clinical' in config.data.supplemental_data.scan():
        pc_file = pd.read_csv(
            config.data.supplemental_data.clinical,
            index_col=0
            )
        prepper.get_pt_chars(pc_file)
    
    prepper.save(
        config.data.processed_data_path,
        boxed=True,
        boxshape=config.data.preprocessing.static.crop,
        level=config.data.preprocessing.static.center_on
        )
            
if __name__ == "__main__":
    run()