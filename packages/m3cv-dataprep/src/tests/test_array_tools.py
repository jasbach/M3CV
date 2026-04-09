from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from m3cv_prep.array_tools import (
    construct_arrays,
    pack_array_sparsely,
    unpack_sparse_array,
)
from m3cv_prep.arrays.exceptions import ROINotFoundError


def test_pack_unpack_sparse_array():
    # Create a sample 3D array
    original_array = np.zeros((4, 4, 4), dtype=int)
    original_array[1, 1, 1] = 1
    original_array[2, 2, 2] = 1
    original_array[3, 3, 3] = 1

    # Pack the array sparsely
    rows, cols, slices = pack_array_sparsely(original_array)

    # Unpack the sparse representation back to a dense array
    unpacked_array = unpack_sparse_array(rows, cols, slices, original_array.shape)

    # Assert that the unpacked array matches the original array
    assert np.array_equal(
        original_array, unpacked_array
    ), "Unpacked array does not match the original"


class TestConstructArraysAliases:
    """Tests for structure_aliases path in construct_arrays."""

    _FOR_UID = "1.2.3.test"

    def _make_grouped(self, has_rtstruct=True, roi_names=None):
        ct_mock = MagicMock()
        ct_mock.FoR = self._FOR_UID
        grouped = {"CT": [ct_mock]}
        if has_rtstruct:
            ssfile = MagicMock()
            roi_items = []
            for name in roi_names or []:
                roi = MagicMock()
                roi.ROIName = name
                roi.ReferencedFrameOfReferenceUID = self._FOR_UID
                roi_items.append(roi)
            ssfile.StructureSetROISequence = roi_items
            grouped["RTSTRUCT"] = [ssfile]
        return grouped

    @patch("m3cv_prep.array_tools.PatientCT")
    @patch("m3cv_prep.array_tools.PatientMask")
    def test_alias_first_match_wins(self, mock_mask_cls, mock_ct_cls):
        """Canonical name is stored as the dict key when first alias matches."""
        mock_ct = MagicMock()
        mock_ct.FoR = self._FOR_UID
        mock_ct_cls.from_dicom_files.return_value = mock_ct
        mock_mask = MagicMock()
        mock_mask_cls.from_rtstruct.return_value = mock_mask

        grouped = self._make_grouped(
            roi_names=["Parotid_L", "parotid_lt", "Lt_Parotid"]
        )
        aliases = {"Parotid_L": ["Parotid_L", "parotid_lt", "Lt_Parotid"]}

        _, _, structure_masks = construct_arrays(grouped, structure_aliases=aliases)

        assert "Parotid_L" in structure_masks
        assert structure_masks["Parotid_L"] is mock_mask
        mock_mask_cls.from_rtstruct.assert_called_once_with(
            reference=mock_ct,
            ssfile=grouped["RTSTRUCT"][0],
            roi_name="Parotid_L",
            proper_name="Parotid_L",
        )

    @patch("m3cv_prep.array_tools.PatientCT")
    @patch("m3cv_prep.array_tools.PatientMask")
    def test_alias_fallthrough_to_second(self, mock_mask_cls, mock_ct_cls):
        """Second alias is tried when first raises ROINotFoundError."""
        mock_ct = MagicMock()
        mock_ct.FoR = self._FOR_UID
        mock_ct_cls.from_dicom_files.return_value = mock_ct
        mock_mask = MagicMock()

        def from_rtstruct_side_effect(**kwargs):
            if kwargs["roi_name"] == "Parotid_L":
                raise ROINotFoundError("Parotid_L")
            return mock_mask

        mock_mask_cls.from_rtstruct.side_effect = from_rtstruct_side_effect

        grouped = self._make_grouped(roi_names=["Parotid_L", "parotid_lt"])
        aliases = {"Parotid_L": ["Parotid_L", "parotid_lt"]}

        _, _, structure_masks = construct_arrays(grouped, structure_aliases=aliases)

        assert structure_masks["Parotid_L"] is mock_mask
        assert mock_mask_cls.from_rtstruct.call_count == 2

    @patch("m3cv_prep.array_tools.PatientCT")
    @patch("m3cv_prep.array_tools.PatientMask")
    def test_alias_no_match_warns(self, mock_mask_cls, mock_ct_cls):
        """UserWarning emitted with canonical name when no alias matches."""
        mock_ct = MagicMock()
        mock_ct.FoR = self._FOR_UID
        mock_ct_cls.from_dicom_files.return_value = mock_ct
        mock_mask_cls.from_rtstruct.side_effect = ROINotFoundError("any")

        grouped = self._make_grouped(roi_names=["GTV", "gtv_primary"])
        aliases = {"GTV": ["GTV", "gtv_primary"]}

        with pytest.warns(UserWarning, match="Skipping structure 'GTV'"):
            construct_arrays(grouped, structure_aliases=aliases)

    def test_both_structure_args_raises(self):
        """ValueError raised when both structure_names and structure_aliases are provided."""
        grouped = self._make_grouped(roi_names=["GTV"])
        with pytest.raises(ValueError, match="mutually exclusive"):
            construct_arrays(
                grouped,
                structure_names=["GTV"],
                structure_aliases={"GTV": ["GTV"]},
            )
