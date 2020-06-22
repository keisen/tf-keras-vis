import numpy as np
import pytest
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten
from tensorflow.keras.models import Sequential, Model

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


@pytest.fixture(scope="function", autouse=True)
def multiple_inputs_cnn_model():
    input_a = Input((8, 8, 3))
    input_b = Input((10, 10, 3))
    x_a = Conv2D(2, 5, activation='relu')(input_a)
    x_b = Conv2D(2, 5, activation='relu')(input_b)
    x = K.concatenate([Flatten()(x_a), Flatten()(x_b)], axis=-1)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=[input_a, input_b], outputs=x)


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


def test__call__if_model_has_multiple_inputs(multiple_inputs_cnn_model):
    gradcam = Gradcam(multiple_inputs_cnn_model)
    result = gradcam(
        SmoothedLoss(1), [np.random.sample(
            (1, 8, 8, 3)), np.random.sample((1, 10, 10, 3))])
    assert len(result) == 2
    assert result[0].shape == (1, 8, 8)
    assert result[1].shape == (1, 10, 10)


def test__call__if_expand_cam_is_False(cnn_model):
    gradcam = Gradcam(cnn_model)
    result = gradcam(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)), expand_cam=False)
    assert result.shape == (1, 6, 6)


def test__call__if_expand_cam_is_False_and_model_has_multiple_inputs(multiple_inputs_cnn_model):
    gradcam = Gradcam(multiple_inputs_cnn_model)
    result = gradcam(
        SmoothedLoss(1), [np.random.sample(
            (1, 8, 8, 3)), np.random.sample((1, 10, 10, 3))],
        expand_cam=False)
    assert result.shape == (1, 6, 6)
