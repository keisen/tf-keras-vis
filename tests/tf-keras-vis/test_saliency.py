import numpy as np
import pytest
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential

from tf_keras_vis.saliency import Saliency
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
def multiple_output_model():
    inputs = Input(shape=(3, ))
    x = Dense(5, activation='relu')(inputs)
    x1 = Dense(1, activation='sigmoid')(x)
    x2 = Dense(1, activation='linear')(x)
    x3 = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=[x1, x2, x3])


def test__call__if_loss_is_None(cnn_model):
    saliency = Saliency(cnn_model)
    try:
        saliency(None, None)
        assert False
    except ValueError:
        assert True


def test__call__if_seed_input_is_None(cnn_model):
    saliency = Saliency(cnn_model)
    try:
        saliency(SmoothedLoss(1), None)
        assert False
    except ValueError:
        assert True


def test__call__if_seed_input_has_not_batch_dim(cnn_model):
    saliency = Saliency(cnn_model)
    result = saliency(SmoothedLoss(1), np.random.sample((8, 8, 3)))
    assert result.shape == (1, 8, 8)


def test__call__(cnn_model):
    saliency = Saliency(cnn_model)
    result = saliency(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)))
    assert result.shape == (1, 8, 8)


def test__call__if_keepdims_is_active(dense_model):
    saliency = Saliency(dense_model)
    result = saliency(SmoothedLoss(1), np.random.sample((3, )), keepdims=True)
    assert result.shape == (1, 3)


def test__call__if_smoothing_is_active(cnn_model):
    saliency = Saliency(cnn_model)
    result = saliency(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)), smooth_samples=1)
    assert result.shape == (1, 8, 8)
    result = saliency(SmoothedLoss(1), np.random.sample((1, 8, 8, 3)), smooth_samples=2)
    assert result.shape == (1, 8, 8)


def test__call__when_model_has_multiple_outputs(multiple_output_model):
    saliency = Saliency(multiple_output_model)

    def loss(output):
        return K.sum(output[0])

    result = saliency([loss, loss, loss], np.random.sample((3, )), keepdims=True)
    assert result.shape == (1, 3)
