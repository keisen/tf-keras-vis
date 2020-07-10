import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tf_keras_vis import ModelVisualization


@pytest.fixture(scope="function", autouse=True)
def model():
    return Sequential([Dense(5, input_shape=(3, )), Dense(2, activation='softmax')])


class MockVisualizer(ModelVisualization):
    def __call__(self):
        pass


def change_activation(model):
    model.layers[-1].activation = tf.keras.activations.linear


def test__init__(model):
    mock = MockVisualizer(model)
    assert mock.model != model
    assert np.array_equal(mock.model.get_weights()[0], model.get_weights()[0])


def test__init__if_clone_is_False(model):
    mock = MockVisualizer(model, clone=False)
    assert mock.model == model


def test__init__if_set_model_modifier(model):
    mock = MockVisualizer(model, change_activation)
    assert mock.model != model
    assert mock.model.layers[-1].activation == tf.keras.activations.linear
    assert model.layers[-1].activation == tf.keras.activations.softmax


def test__init__if_set_model_modifier_that_return_other_model(model):
    another_model = Sequential([Dense(5, input_shape=(3, ))])
    mock = MockVisualizer(model, lambda m: another_model)
    assert mock.model != model
    assert mock.model == another_model
