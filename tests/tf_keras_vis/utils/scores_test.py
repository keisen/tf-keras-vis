import numpy as np
import pytest
import tensorflow as tf

from tf_keras_vis.utils.scores import BinaryScore, CategoricalScore, InactiveScore
from tf_keras_vis.utils.test import NO_ERROR, assert_raises, dummy_sample


class TestInactiveScore():
    @pytest.mark.parametrize("output,expected_shape,expected_error", [
        (dummy_sample((1, 1)), (1, 1), NO_ERROR),
        (dummy_sample((10, 5)), (10, 5), NO_ERROR),
        (dummy_sample((1, 224, 224, 3)), (1, 224, 224, 3), NO_ERROR),
    ])
    def test__call__(self, output, expected_shape, expected_error):
        with assert_raises(expected_error):
            actual = InactiveScore()(output)
            assert np.all(actual == 0.0)
            assert actual.shape == expected_shape


class TestBinaryScore():
    @pytest.mark.parametrize("target_values,expected,expected_error", [
        (None, None, ValueError),
        (0, [False], NO_ERROR),
        (1, [True], NO_ERROR),
        (100, [True], NO_ERROR),
        (-1, [True], NO_ERROR),
        (1.0, [True], NO_ERROR),
        ([], None, ValueError),
        ([None], None, ValueError),
        ([0, 0], [False, False], NO_ERROR),
        ([0, 1, 0], [False, True, False], NO_ERROR),
        ([-1, 0], [True, False], NO_ERROR),
    ])
    def test__init__(self, target_values, expected, expected_error):
        with assert_raises(expected_error):
            score = BinaryScore(target_values)
            assert score.target_values == expected

    @pytest.mark.parametrize("target_values,output,expected,expected_error", [
        (False, [[1, 1, 0], [1, 0, 1]], [-1], ValueError),
        (False, [[1]], [-1], NO_ERROR),
        (False, [[0]], [0], NO_ERROR),
        (True, [[1]], [1], NO_ERROR),
        (True, [[0]], [0], NO_ERROR),
        (True, [[0], [1], [0]], [0, 1, 0], NO_ERROR),
        (False, [[0], [1], [0]], [0, -1, 0], NO_ERROR),
        ([True, False, True], [[0], [1], [0]], [0, -1, 0], NO_ERROR),
        ([False, True, False], [[0], [1], [0]], [0, 1, 0], NO_ERROR),
    ])
    def test__call__(self, target_values, output, expected, expected_error):
        output = tf.constant(output, tf.float32)
        score = BinaryScore(target_values)
        with assert_raises(expected_error):
            score_value = score(output)
            assert tf.math.reduce_all(score_value == expected)


class TestCategoricalScore():
    @pytest.mark.parametrize("indices,expected,expected_error", [
        (None, None, ValueError),
        (5, [5], NO_ERROR),
        ((1, ), [1], NO_ERROR),
        ([3], [3], NO_ERROR),
        ([], None, ValueError),
        ([None], None, ValueError),
        ([2, None], None, ValueError),
        ((0, 8, 3), [0, 8, 3], NO_ERROR),
        ([0, 8, 3], [0, 8, 3], NO_ERROR),
    ])
    def test__init__(self, indices, expected, expected_error):
        with assert_raises(expected_error):
            score = CategoricalScore(indices)
            assert score.indices == expected

    @pytest.mark.parametrize("indices,output_shape,expected_error", [
        (2, (1, ), ValueError),
        (2, (1, 2), ValueError),
        (2, (1, 4, 1), ValueError),
        (2, (1, 4, 3), NO_ERROR),
        (2, (2, 4, 3), NO_ERROR),
        (2, (8, 32, 32, 3), NO_ERROR),
    ])
    def test__call__(self, indices, output_shape, expected_error):
        output = tf.constant(dummy_sample(output_shape), tf.float32)
        score = CategoricalScore(indices)
        with assert_raises(expected_error):
            score_value = score(output)
            assert score_value.shape == output_shape[0:1]
