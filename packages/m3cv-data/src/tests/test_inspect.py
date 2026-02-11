"""Tests for inspect.py module."""

from __future__ import annotations

import pytest

from m3cv_data import inspect_directory, inspect_h5


class TestInspectH5:
    """Tests for inspect_h5 function."""

    def test_inspect_full_file(self, sample_h5_path: str) -> None:
        """Test inspecting a file with all data."""
        info = inspect_h5(sample_h5_path)

        assert info.patient_id == "TEST001"
        assert info.has_ct is True
        assert info.has_dose is True
        assert info.ct_shape == (10, 64, 64)
        assert "GTV" in info.structure_names
        assert "PTV" in info.structure_names
        assert info.file_size_mb > 0

    def test_inspect_minimal_file(self, sample_h5_minimal: str) -> None:
        """Test inspecting a file with only CT."""
        info = inspect_h5(sample_h5_minimal)

        assert info.patient_id == "MINIMAL001"
        assert info.has_ct is True
        assert info.has_dose is False
        assert info.ct_shape == (5, 32, 32)
        assert info.structure_names == []

    def test_inspect_nonexistent_file(self) -> None:
        """Test that inspecting nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            inspect_h5("/nonexistent/path.h5")

    def test_info_repr(self, sample_h5_path: str) -> None:
        """Test H5FileInfo __repr__."""
        info = inspect_h5(sample_h5_path)
        repr_str = repr(info)
        assert "TEST001" in repr_str
        assert "ct=True" in repr_str
        assert "dose=True" in repr_str


class TestInspectDirectory:
    """Tests for inspect_directory function."""

    def test_inspect_directory_basic(self, sample_h5_directory: str) -> None:
        """Test inspecting a directory of HDF5 files."""
        infos = inspect_directory(sample_h5_directory)

        assert len(infos) == 3
        # Should be sorted by filename
        assert infos[0].patient_id == "PATIENT000"
        assert infos[1].patient_id == "PATIENT001"
        assert infos[2].patient_id == "PATIENT002"

    def test_inspect_directory_with_pattern(
        self, sample_h5_directory: str, tmp_path
    ) -> None:
        """Test inspecting with specific pattern."""
        # Create a non-h5 file
        (tmp_path / "not_h5.txt").write_text("test")

        infos = inspect_directory(sample_h5_directory, pattern="*.h5")
        assert len(infos) == 3

    def test_inspect_directory_not_a_directory(self, sample_h5_path: str) -> None:
        """Test that inspecting a file raises NotADirectoryError."""
        with pytest.raises(NotADirectoryError):
            inspect_directory(sample_h5_path)

    def test_inspect_directory_empty(self, tmp_path) -> None:
        """Test inspecting empty directory returns empty list."""
        infos = inspect_directory(str(tmp_path))
        assert infos == []


class TestSummaryTable:
    """Tests for summary_table function."""

    def test_summary_table_empty(self, capsys) -> None:
        """Test summary_table with empty list."""
        from m3cv_data import summary_table

        summary_table([])
        captured = capsys.readouterr()
        assert "No files" in captured.out

    def test_summary_table_with_data(self, sample_h5_directory: str, capsys) -> None:
        """Test summary_table prints output."""
        from m3cv_data import summary_table

        infos = inspect_directory(sample_h5_directory)
        summary_table(infos)
        captured = capsys.readouterr()

        # Should print something
        assert len(captured.out) > 0
        # Should include patient count
        assert "3" in captured.out
