"""Frame alignment and conservation checks for the report's map reconstruction."""

import numpy as np
import pytest

from harvest.coordination import reconstruct_counts


def test_reconstruction_includes_initial_and_final_states():
    events = [(1, 1, 2), (3, 3, 1)]
    actual = reconstruct_counts([2, 1, 1], events, horizon=3)
    np.testing.assert_array_equal(actual, [[2, 1, 1], [1, 2, 1], [1, 2, 1], [2, 2, 0]])
    assert np.all(actual.sum(axis=1) == 4)


def test_multiple_changes_in_one_frame_are_accumulated():
    actual = reconstruct_counts([2, 1, 1], [(1, 1, 2), (1, 2, 3)], horizon=1)
    np.testing.assert_array_equal(actual[-1], [1, 1, 2])


@pytest.mark.parametrize("event", [(0, 1, 2), (3, 1, 2), (1, 0, 2), (1, 1, 4)])
def test_invalid_event_is_rejected(event):
    with pytest.raises(ValueError):
        reconstruct_counts([1, 1, 1], [event], horizon=2)


def test_negative_reconstructed_count_is_rejected():
    with pytest.raises(ValueError):
        reconstruct_counts([1, 1, 1], [(1, 1, 2), (1, 1, 2)], horizon=1)
