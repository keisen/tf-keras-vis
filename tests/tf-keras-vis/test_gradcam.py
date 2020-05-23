import numpy as np
import pytest
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.losses import SmoothedLoss


@pytest.fixture(scope="function", autouse=True)
def dense_model():
    return Sequential(
        [Dense(5, input_shape=(3, ), activation='relu'),
         Dense(2, activation='softmax')])


@pytest.fixture(scope="function", autouse=True)
def cnn_model():
    return Sequential([
        Conv2D(5, 3, input_shape=(8, 8, 3), activation='relu'),
        Flatten(),
        Dense(2, activation='softmax')
    ])


def test__call__if_loss_is_None(cnn_model):
    gradcam = Gradcam(cnn_model)
    try:
        gradcam(None, None)
        assert False
    except ValueError:
        assert True


def test__call__if_seed_input_is_None(cnn_model):
    gradcam = Gradcam(cnn_model)
    try:
        gradcam(SmoothedLoss(1), None)
        assert False
    except ValueError:
        assert True


def test__call__if_seed_input_has_not_batch_dim(cnn_model):
    gradcam = Gradcam(cnn_model)
    result = gradcam(SmoothedLoss(1), np.random.sample((8, 8, 3)))
    assert result.shape == (1, 8, 8)


def test__call__(cnn_model):
    gradcam = Gradcam(cnn_model)
    result = gradcam(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)))
    assert result.shape == (1, 8, 8)


def test__call__if_penultimate_layer_is_None(cnn_model):
    gradcam = Gradcam(cnn_model)
    result = gradcam(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)), penultimate_layer=None)
    assert result.shape == (1, 8, 8)


def test__call__if_penultimate_layer_is_noexist_index(cnn_model):
    gradcam = Gradcam(cnn_model)
    try:
        gradcam(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)), penultimate_layer=100000)
        assert False
    except ValueError:
        assert True


def test__call__if_penultimate_layer_is_noexist_name(cnn_model):
    gradcam = Gradcam(cnn_model)
    try:
        gradcam(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)), penultimate_layer='hoge')
        assert False
    except ValueError:
        assert True


def test__call__if_model_has_only_dense_layer(dense_model):
    gradcam = Gradcam(dense_model)
    result = gradcam(SmoothedLoss(1),
                     np.random.sample((1, 8, 8, 3)),
                     seek_penultimate_conv_layer=False)
    assert result.shape == (1, 8, 8)
    try:
        gradcam(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)))
        assert False
    except ValueError:
        assert True
