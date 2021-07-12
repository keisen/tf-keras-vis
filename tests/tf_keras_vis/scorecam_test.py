import numpy as np
import pytest
import tensorflow as tf

from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.test import (NO_ERROR, assert_raises, dummy_sample, score_with_list,
                                     score_with_tensor, score_with_tuple)


class TestScorecam():
    @pytest.mark.parametrize("max_N,expected_error", [
        (-100, NO_ERROR),
        (-1, NO_ERROR),
        (0, NO_ERROR),
        (1, NO_ERROR),
        (3, NO_ERROR),
        (100, ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_max_N_is_(self, max_N, expected_error, conv_model):
        with assert_raises(expected_error):
            cam = Scorecam(conv_model)
            result = cam(CategoricalScore(0), dummy_sample((2, 8, 8, 3)), max_N=max_N)
            assert result.shape == (2, 8, 8)

    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        (CategoricalScore(0), NO_ERROR),
        (score_with_tuple, NO_ERROR),
        (score_with_list, NO_ERROR),
        (score_with_tensor, NO_ERROR),
        (lambda x: np.mean(x), ValueError),
        (lambda x: tf.reshape(x, (-1, )), ValueError),
        ([None], ValueError),
        ([CategoricalScore(0)], NO_ERROR),
        ([score_with_tuple], NO_ERROR),
        ([score_with_list], NO_ERROR),
        ([score_with_tensor], NO_ERROR),
        ([lambda x: np.mean(x)], ValueError),
        ([lambda x: tf.reshape(x, (-1, ))], ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_score_is_(self, scores, expected_error, conv_model):
        cam = Scorecam(conv_model)
        with assert_raises(expected_error):
            result = cam(scores, dummy_sample((2, 8, 8, 3)))
            assert result.shape == (2, 8, 8)
