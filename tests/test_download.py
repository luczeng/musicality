"""Tests for the download_dataset CLI tool."""

from unittest.mock import patch, MagicMock
from click.testing import CliRunner

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from download_dataset import main

CONFIG_CONTENT = "data_home: {data_home}\ndatasets:\n  - ballroom\n  - gtzan\n"


def _make_config(tmp_path, data_home=None):
    config = tmp_path / "download.yaml"
    config.write_text(CONFIG_CONTENT.format(data_home=data_home or str(tmp_path / "data")))
    return config


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
@patch("download_dataset.mirdata.initialize")
def test_download_single_dataset(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, ["ballroom", "--data-home", str(tmp_path)])

    assert result.exit_code == 0
    mock_init.assert_called_once_with("ballroom", data_home=str(tmp_path / "ballroom"))
    mock_ds.download.assert_called_once()


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
@patch("download_dataset.mirdata.initialize")
def test_download_all_reads_config(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, ["--all"])

    assert result.exit_code == 0
    assert mock_init.call_count == 2


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
@patch("download_dataset.mirdata.initialize")
def test_data_home_from_config(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds
    data_home = tmp_path / "custom_data"

    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path, data_home)):
        result = runner.invoke(main, ["ballroom"])

    assert result.exit_code == 0
    mock_init.assert_called_once_with("ballroom", data_home=str(data_home / "ballroom"))


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
@patch("download_dataset.mirdata.initialize")
def test_skips_already_present_dataset(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    dataset_dir = tmp_path / "ballroom"
    dataset_dir.mkdir()
    (dataset_dir / "somefile.mp3").write_text("existing")

    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, ["ballroom", "--data-home", str(tmp_path)])

    assert result.exit_code == 0
    assert "Skipping" in result.output
    mock_ds.download.assert_not_called()


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
@patch("download_dataset.mirdata.initialize")
def test_download_all_skips_present_downloads_missing(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    # ballroom already present
    ballroom_dir = tmp_path / "data" / "ballroom"
    ballroom_dir.mkdir(parents=True)
    (ballroom_dir / "somefile.mp3").write_text("existing")

    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, ["--all"])

    assert result.exit_code == 0
    assert mock_init.call_count == 1
    mock_init.assert_called_once_with("gtzan", data_home=str(tmp_path / "data" / "gtzan"))


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom"])
def test_unknown_dataset_exits_with_error(mock_list, tmp_path):
    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, ["unknown_ds", "--data-home", str(tmp_path)])

    assert result.exit_code == 1


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom", "gtzan"])
def test_list_flag_prints_datasets(mock_list, tmp_path):
    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, ["--list"])

    assert result.exit_code == 0
    assert "ballroom" in result.output
    assert "gtzan" in result.output


def test_no_args_exits_with_error(tmp_path):
    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, [])

    assert result.exit_code == 1


@patch("download_dataset.mirdata.list_datasets", return_value=["ballroom"])
@patch("download_dataset.mirdata.initialize")
def test_archives_deleted_after_download(mock_init, mock_list, tmp_path):
    mock_ds = MagicMock()
    mock_init.return_value = mock_ds

    dataset_dir = tmp_path / "ballroom"
    archive = dataset_dir / "data.tar.gz"

    def fake_download():
        dataset_dir.mkdir(exist_ok=True)
        archive.write_text("fake")

    mock_ds.download.side_effect = fake_download

    runner = CliRunner()
    with patch("download_dataset.CONFIG_PATH", _make_config(tmp_path)):
        result = runner.invoke(main, ["ballroom", "--data-home", str(tmp_path)])

    assert result.exit_code == 0
    assert not archive.exists()
