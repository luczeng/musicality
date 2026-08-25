"""Lightweight tests for tools.sweep_beat_postprocess's scoring helpers —
``score_combo`` (beat-detection) and ``score_phase_combo`` (bar-position).
DSP (``pick_peaks``/``gate_periodicity``/``label_bar_position``) and
mir_eval scoring are mocked, so these only exercise the tool's own
orchestration/averaging logic, not the postprocessing math itself (covered
by tests/test_postprocess.py and tests/test_metrics.py).
"""

import math
from unittest.mock import patch

import numpy as np
import pytest

from tools.sweep_beat_postprocess import score_combo, score_phase_combo


class TestScoreCombo:
    def test_extracts_beat_channel_from_2d_probs(self):
        cached = [(np.array([0.5]), None, False, np.zeros((3, 10)))]

        with (
            patch(
                "tools.sweep_beat_postprocess.readout_beat_only",
                return_value=np.array([0.5]),
            ) as readout_beat_only,
            patch("tools.sweep_beat_postprocess.beat_f_measure", return_value=1.0),
        ):
            score = score_combo(
                cached,
                fps=43.0,
                beat_threshold=0.3,
                min_distance_frames=1,
                gate_tolerance=0.2,
                tolerance=0.07,
                trim=True,
            )

            probs_passed = readout_beat_only.call_args.args[0]
            assert probs_passed.ndim == 1
            assert score == 1.0

    def test_passes_1d_probs_through_unchanged(self):
        cached = [(np.array([0.5]), None, False, np.zeros(10))]

        with (
            patch(
                "tools.sweep_beat_postprocess.readout_beat_only",
                return_value=np.array([0.5]),
            ) as readout_beat_only,
            patch("tools.sweep_beat_postprocess.beat_f_measure", return_value=1.0),
        ):
            score_combo(
                cached,
                fps=43.0,
                beat_threshold=0.3,
                min_distance_frames=1,
                gate_tolerance=0.2,
                tolerance=0.07,
                trim=True,
            )

            probs_passed = readout_beat_only.call_args.args[0]
            assert probs_passed.shape == (10,)

    def test_averages_across_tracks(self):
        cached = [
            (np.array([0.5]), None, False, np.zeros(10)),
            (np.array([1.5]), None, False, np.zeros(10)),
        ]

        with (
            patch(
                "tools.sweep_beat_postprocess.readout_beat_only",
                return_value=np.array([]),
            ),
            patch(
                "tools.sweep_beat_postprocess.beat_f_measure",
                side_effect=[0.2, 0.8],
            ),
        ):
            score = score_combo(
                cached,
                fps=43.0,
                beat_threshold=0.3,
                min_distance_frames=1,
                gate_tolerance=0.2,
                tolerance=0.07,
                trim=True,
            )

            assert score == 0.5


class TestScorePhaseCombo:
    def _cached(self, has_positions_list):
        return [
            (np.array([0.5]), np.array([1]), has_positions, np.zeros((3, 10)))
            for has_positions in has_positions_list
        ]

    def _kwargs(self):
        return dict(
            fps=43.0,
            beat_threshold=0.3,
            min_distance_frames=1,
            gate_tolerance=0.2,
            anchor_threshold=0.5,
            group_size=4,
            tolerance=0.07,
            trim=True,
        )

    def test_skips_tracks_without_positions(self):
        cached = self._cached([True, False])

        with (
            patch("tools.sweep_beat_postprocess.readout", return_value=[]) as readout,
            patch(
                "tools.sweep_beat_postprocess.downbeat_f_measures",
                return_value=(0.5, 0.6),
            ),
            patch(
                "tools.sweep_beat_postprocess.confusion_half_cycle_rate",
                return_value=0.1,
            ),
        ):
            f_one, f_last, confusion = score_phase_combo(cached, **self._kwargs())

            assert readout.call_count == 1
            assert f_one == 0.5
            assert f_last == 0.6
            assert confusion == 0.1

    def test_filters_none_confusion_but_keeps_f_one_f_last(self):
        cached = self._cached([True, True])

        with (
            patch("tools.sweep_beat_postprocess.readout", return_value=[]),
            patch(
                "tools.sweep_beat_postprocess.downbeat_f_measures",
                side_effect=[(0.2, 0.4), (0.6, 0.8)],
            ),
            patch(
                "tools.sweep_beat_postprocess.confusion_half_cycle_rate",
                side_effect=[None, 0.3],
            ),
        ):
            f_one, f_last, confusion = score_phase_combo(cached, **self._kwargs())

            assert f_one == pytest.approx(0.4)
            assert f_last == pytest.approx(0.6)
            assert confusion == 0.3

    def test_returns_nan_when_no_tracks_have_positions(self):
        cached = self._cached([False, False])

        f_one, f_last, confusion = score_phase_combo(cached, **self._kwargs())

        assert math.isnan(f_one)
        assert math.isnan(f_last)
        assert math.isnan(confusion)
