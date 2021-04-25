import pytest
import torch

from ..test_metrics.test_distance.test_functional import LEV_PARAMETRIZE

import vak.nn.functional as F


def _str_to_tensor(_str):
    return torch.tensor([ord(char) for char in _str])


LEV_PARAMETRIZE = [
    (_str_to_tensor(source), _str_to_tensor(target), dist)
    for (source, target, dist) in LEV_PARAMETRIZE
]


@pytest.mark.parametrize(
    "source, target, expected_distance",
    LEV_PARAMETRIZE
)
def test_levenshtein(source, target, expected_distance):
    assert F.levenshtein(source, target).item() == expected_distance


@pytest.mark.parametrize(
    "source, target, expected_distance",
    LEV_PARAMETRIZE
)
def test_segment_error_rate(source, target, expected_distance):
    expected_rate = torch.tensor(expected_distance) / torch.numel(target)
    assert F.segment_error_rate(source, target) == expected_rate
