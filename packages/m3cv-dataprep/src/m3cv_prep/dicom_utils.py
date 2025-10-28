import pydicom
from pydicom.dataset import Dataset

def validate_patientid(dcms):
    """
    Validate the loaded DICOM files to ensure they all belong to the same patient.

    Args:
        dcms (list): List of pydicom Dataset objects.
    Raises:
        ValueError: If the DICOM files belong to multiple patients.
    """
    if not dcms:
        return
    patient_ids = {dcm.PatientID for dcm in dcms}
    if len(patient_ids) > 1:
        raise ValueError("DICOM files belong to multiple patients.")
    
def validate_fields(dcms: list[Dataset], fields: list[str]):
    """
    Ensures that all DICOM files in the list share the same value for specified attributes.
    """
    core_fields = ['PatientID', 'StudyInstanceUID']
    fields = list(set(fields) | set(core_fields))
    errors = []
    for field in fields:
        first_value = getattr(dcms[0], field, None)
        for dcm in dcms[1:]:
            if getattr(dcm, field, None) != first_value:
                errors.append(f"DICOM files do not share the same value for attribute '{field}'.")
                break
    if errors:
        raise ValueError(" \n".join(errors))
    
def group_dcms_by_modality(dcms: list[Dataset]) -> dict[str, list[Dataset]]:
    """
    Sort DICOM files by their Modality attribute.

    Args:
        dcms (list): List of pydicom Dataset objects.
    Returns:
        dict: A dictionary mapping modality strings to lists of DICOM Dataset objects.
    """
    modality_dict = {}
    for dcm in dcms:
        modality = dcm.Modality
        if modality not in modality_dict:
            modality_dict[modality] = []
        modality_dict[modality].append(dcm)
    return modality_dict