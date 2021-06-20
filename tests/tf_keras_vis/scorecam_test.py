import pytest

from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.test import NO_ERROR, assert_error, dummy_sample


@pytest.mark.parametrize("max_N,expected_error", [
    (-100, NO_ERROR),
    (-1, NO_ERROR),
    (0, NO_ERROR),
    (1, NO_ERROR),
    (3, NO_ERROR),
    (100, ValueError),
])
@pytest.mark.usefixtures("mixed_precision")
class TestScorecam():
    def test__call__if_max_N_is_(self, max_N, expected_error, conv_model):
        with assert_error(expected_error):
            cam = Scorecam(conv_model)
            result = cam(CategoricalScore(0), dummy_sample((1, 8, 8, 3)), max_N=max_N)
            assert result.shape == (1, 8, 8)
