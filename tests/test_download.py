"""Tests for the download_dataset CLI tool."""

from unittest.mock import patch, MagicMock
from click.testing import CliRunner

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from download_dataset import main


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
@patch("download_dataset.mirdata.initialize")
def test_download_single_dataset(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    runner = CliRunner()
    result = runner.invoke(main, ["ballroom", "--data-home", str(tmp_path)])

    assert result.exit_code == 0
    mock_init.assert_called_once_with("ballroom", data_home=str(tmp_path / "ballroom"))
    mock_ds.download.assert_called_once()


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
@patch("download_dataset.mirdata.initialize")
def test_download_all_reads_config(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    config = tmp_path / "download.yaml"
    config.write_text("datasets:\n  - ballroom\n  - gtzan\n")

    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", config):
        result = runner.invoke(main, ["--all", "--data-home", str(tmp_path)])

    assert result.exit_code == 0
    assert mock_init.call_count == 2


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom"])
def test_unknown_dataset_exits_with_error(mock_list, tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["unknown_ds", "--data-home", str(tmp_path)])

    assert result.exit_code == 1


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
def test_list_flag_prints_datasets(mock_list):
    runner = CliRunner()
    result = runner.invoke(main, ["--list"])

    assert result.exit_code == 0
    assert "ballroom" in result.output
    assert "gtzan" in result.output


def test_no_args_exits_with_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["--data-home", str(tmp_path)])

    assert result.exit_code == 1


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom"])
@patch("download_dataset.mirdata.initialize")
def test_archives_deleted_after_download(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    dataset_dir = tmp_path / "ballroom"
    dataset_dir.mkdir()
    archive = dataset_dir / "data.tar.gz"
    archive.write_text("fake")

    def fake_download():
        pass  # archive already exists, no-op

    mock_ds.download.side_effect = fake_download

    runner = CliRunner()
    result = runner.invoke(main, ["ballroom", "--data-home", str(tmp_path)])

    assert result.exit_code == 0
    assert not archive.exists()
