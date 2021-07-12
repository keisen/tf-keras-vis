import pytest
import tensorflow as tf

from tf_keras_vis.activation_maximization.input_modifiers import Rotate, Rotate2D, Scale
from tf_keras_vis.utils.test import NO_ERROR, assert_raises, dummy_sample


class TestRotate():
    @pytest.mark.parametrize("degree", [0, 1, 3, 0.0, 1.0, 3.0])
    @pytest.mark.parametrize("axes,expected_error", [
        (None, NO_ERROR),
        ((None, ), ValueError),
        ((0, ), ValueError),
        ((0, 1), NO_ERROR),
        ([0, 1], NO_ERROR),
        ((0.0, 1.0), TypeError),
        ((0, 1, 2), ValueError),
    ])
    def test__init__(self, degree, axes, expected_error):
        with assert_raises(expected_error):
            if axes is None:
                instance = Rotate(degree=degree)
            else:
                instance = Rotate(axes=axes, degree=degree)
                assert instance.axes == axes
            assert instance.degree == float(degree)


class TestRotate2D():
    @pytest.mark.parametrize("degree", [0, 1, 3, 0.0, 1.0, 3.0])
    def test__init__(self, degree):
        instance = Rotate2D(degree=degree)
        assert instance.axes == (1, 2)
        assert instance.degree == float(degree)


class TestScale():
    @pytest.mark.parametrize(
        "seed_input", [dummy_sample(
            (1, 8, 8, 3)), tf.constant(dummy_sample((1, 8, 8, 3)))])
    def test__call__(self, seed_input):
        result = Scale()(seed_input)
        assert result.shape == seed_input.shape
