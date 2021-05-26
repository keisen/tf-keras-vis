import numpy as np
import pytest
import tensorflow as tf

from tf_keras_vis.utils.scores import (BinaryScore, CategoricalScore,
                                       InactiveScore)
from tf_keras_vis.utils.test import does_not_raise, dummy_sample


class TestInactiveScore():
    @pytest.mark.parametrize("output,expected_shape,expectation", [
        (dummy_sample((1, 1)), (1, 1), does_not_raise()),
        (dummy_sample((10, 5)), (10, 5), does_not_raise()),
        (dummy_sample((1, 224, 224, 3)), (1, 224, 224, 3), does_not_raise()),
    ])
    def test__call__(self, output, expected_shape, expectation):
        with expectation:
            actual = InactiveScore()(output)
            assert np.all(actual == 0.0)
            assert actual.shape == expected_shape


class TestBinaryScore():
    @pytest.mark.parametrize("target_values,expected,expectation", [
        (None, None, pytest.raises(ValueError)),
        (0, [False], does_not_raise()),
        (1, [True], does_not_raise()),
        (100, [True], does_not_raise()),
        (-1, [True], does_not_raise()),
        (1.0, [True], does_not_raise()),
        ([None], None, pytest.raises(ValueError)),
        ([0, 0], [False, False], does_not_raise()),
        ([0, 1, 0], [False, True, False], does_not_raise()),
        ([-1, 0], [True, False], does_not_raise()),
    ])
    def test__init__(self, target_values, expected, expectation):
        with expectation:
            score = BinaryScore(target_values)
            assert score.target_values == expected

    @pytest.mark.parametrize("target_values,output,expected,expectation", [
        (False, [[1, 1, 0], [1, 0, 1]], [0], pytest.raises(ValueError)),
        (False, [[1]], [0], does_not_raise()),
        (False, [[0]], [1], does_not_raise()),
        (True, [[1]], [1], does_not_raise()),
        (True, [[0]], [0], does_not_raise()),
        (True, [[0], [1], [0]], [0, 1, 0], does_not_raise()),
        (False, [[0], [1], [0]], [1, 0, 1], does_not_raise()),
        ([True, False, True], [[0], [1], [0]], [0, 0, 0], does_not_raise()),
        ([False, True, False], [[0], [1], [0]], [1, 1, 1], does_not_raise()),
    ])
    def test__call__(self, target_values, output, expected, expectation):
        output = tf.constant(output, tf.float32)
        score = BinaryScore(target_values)
        with expectation:
            score_value = score(output)
            assert score_value == expected


class TestCategoricalScore():
    @pytest.mark.parametrize("indices,expected,expectation", [
        (None, None, pytest.raises(ValueError)),
        (5, [5], does_not_raise()),
        ((1, ), [1], does_not_raise()),
        ([3], [3], does_not_raise()),
        ([None], None, pytest.raises(ValueError)),
        ([2, None], None, pytest.raises(ValueError)),
        ((0, 8, 3), [0, 8, 3], does_not_raise()),
        ([0, 8, 3], [0, 8, 3], does_not_raise()),
    ])
    def test__init__(self, indices, expected, expectation):
        with expectation:
            score = CategoricalScore(indices)
            assert score.indices == expected

    @pytest.mark.parametrize("indices,output_shape,expectation", [
        (2, (1, ), pytest.raises(ValueError)),
        (2, (1, 2), pytest.raises(ValueError)),
        (2, (1, 4, 1), pytest.raises(ValueError)),
        (2, (1, 4, 3), does_not_raise()),
        (2, (2, 4, 3), does_not_raise()),
        (2, (8, 32, 32, 3), does_not_raise()),
    ])
    def test__call__(self, indices, output_shape, expectation):
        output = tf.constant(dummy_sample(output_shape), tf.float32)
        score = CategoricalScore(indices)
        with expectation:
            score_value = score(output)
            assert score_value.shape == output_shape[0:1]
