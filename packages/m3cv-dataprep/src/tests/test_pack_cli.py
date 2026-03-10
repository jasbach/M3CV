"""Tests for pack CLI alias-file handling."""

from __future__ import annotations

import json
import os

import pytest
import typer

from m3cv_prep.cli.pack import _load_alias_file


class TestLoadAliasFile:
    """Tests for _load_alias_file validation."""

    def _write_json(self, tmp_path, data):
        path = os.path.join(tmp_path, "aliases.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_valid_alias_file(self, tmp_path):
        data = {
            "Parotid_L": ["Parotid_L", "parotid_lt", "Lt_Parotid"],
            "GTV": ["GTV", "gtv_primary"],
        }
        path = self._write_json(tmp_path, data)
        result = _load_alias_file(path)
        assert result == data

    def test_file_not_found_raises(self):
        with pytest.raises(typer.BadParameter, match="not found"):
            _load_alias_file("/nonexistent/path/aliases.json")

    def test_invalid_json_raises(self, tmp_path):
        path = os.path.join(tmp_path, "bad.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        with pytest.raises(typer.BadParameter, match="Invalid JSON"):
            _load_alias_file(path)

    def test_top_level_not_dict_raises(self, tmp_path):
        path = self._write_json(tmp_path, ["Parotid_L", "GTV"])
        with pytest.raises(typer.BadParameter, match="JSON object"):
            _load_alias_file(path)

    def test_value_not_list_raises(self, tmp_path):
        path = self._write_json(tmp_path, {"Parotid_L": "parotid_lt"})
        with pytest.raises(typer.BadParameter, match="non-empty lists"):
            _load_alias_file(path)

    def test_empty_list_raises(self, tmp_path):
        path = self._write_json(tmp_path, {"Parotid_L": []})
        with pytest.raises(typer.BadParameter, match="non-empty lists"):
            _load_alias_file(path)

    def test_alias_not_string_raises(self, tmp_path):
        path = self._write_json(tmp_path, {"Parotid_L": ["valid", 42]})
        with pytest.raises(typer.BadParameter, match="strings"):
            _load_alias_file(path)


class TestPackMutualExclusivity:
    """Test that --structures and --alias-file are mutually exclusive."""

    def test_both_options_raise(self, tmp_path):
        alias_path = os.path.join(tmp_path, "aliases.json")
        with open(alias_path, "w") as f:
            json.dump({"GTV": ["GTV"]}, f)

        import typer as _typer
        from typer.testing import CliRunner

        from m3cv_prep.cli.pack import pack

        app = _typer.Typer()
        app.command()(pack)
        runner = CliRunner()

        result = runner.invoke(
            app,
            [str(tmp_path), "--structures", "GTV", "--alias-file", alias_path],
        )
        assert result.exit_code != 0
        assert (
            "mutually exclusive" in result.output.lower()
            or result.exception is not None
        )
