import numpy as np
import pytest
import tensorflow as tf

from tf_keras_vis import ModelVisualization
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.test import dummy_sample


class MockVisualizer(ModelVisualization):
    def __call__(self):
        pass


class TestModelVisualization():
    @pytest.mark.parametrize("modifier,clone,expected_same,expected_activation", [
        (None, False, True, tf.keras.activations.softmax),
        (None, True, True, tf.keras.activations.softmax),
        ('not-return', False, True, tf.keras.activations.linear),
        ('not-return', True, False, tf.keras.activations.linear),
        ('return', False, True, tf.keras.activations.linear),
        ('return', True, False, tf.keras.activations.linear),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__init__(self, modifier, clone, expected_same, expected_activation, conv_model):
        if modifier == 'return':
            mock = MockVisualizer(conv_model, model_modifier=ReplaceToLinear(), clone=clone)
        elif modifier == 'not-return':
            mock = MockVisualizer(conv_model, model_modifier=ReplaceToLinear(), clone=clone)
        else:
            mock = MockVisualizer(conv_model, clone=clone)
        assert (mock.model is conv_model) == expected_same
        assert mock.model.layers[-1].activation == expected_activation
        assert np.array_equal(mock.model.get_weights()[0], conv_model.get_weights()[0])

    @pytest.mark.parametrize("score,expected_shape", [
        (dummy_sample((2, 32, 32, 3)), (2, )),
        ((dummy_sample((32, 32, 3)), dummy_sample((32, 32, 3))), (2, )),
        ([dummy_sample((32, 32, 3)), dummy_sample((32, 32, 3))], (2, )),
        (tf.constant(dummy_sample((2, 32, 32, 3))), (2, )),
        ((tf.constant(dummy_sample((32, 32, 3))), tf.constant(dummy_sample((32, 32, 3)))), (2, )),
        ([tf.constant(dummy_sample((32, 32, 3))),
          tf.constant(dummy_sample((32, 32, 3)))], (2, )),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test_mean_score_value(self, score, expected_shape, conv_model):
        actual = MockVisualizer(conv_model)._mean_score_value(score)
        assert actual.shape == expected_shape
