import numpy as np
import pytest
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, Dense, GlobalAveragePooling2D, Input)

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore


@pytest.fixture(scope="function", autouse=True)
def dense_model():
    inputs = Input((3, ))
    x = Dense(5, activation='relu')(inputs)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)


@pytest.fixture(scope="function", autouse=True)
def model():
    inputs = Input((8, 8, 3))
    x = Conv2D(2, 3, activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)


@pytest.fixture(scope="function", autouse=True)
def multiple_inputs_model():
    inputs1 = Input((8, 8, 3))
    inputs2 = Input((10, 10, 3))
    x1 = Conv2D(2, 3, padding='same', activation='relu')(inputs1)
    x2 = Conv2D(2, 3, activation='relu')(inputs2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=[inputs1, inputs2], outputs=x)


@pytest.fixture(scope="function", autouse=True)
def multiple_outputs_model():
    inputs = Input((8, 8, 3))
    x = Conv2D(2, 3, activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, activation='softmax')(x)
    x2 = Dense(1)(x)
    return Model(inputs=inputs, outputs=[x1, x2])


@pytest.fixture(scope="function", autouse=True)
def multiple_io_model():
    inputs1 = Input((8, 8, 3))
    inputs2 = Input((10, 10, 3))
    x1 = Conv2D(2, 3, padding='same', activation='relu')(inputs1)
    x2 = Conv2D(2, 3, activation='relu')(inputs2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, activation='softmax')(x)
    x2 = Dense(1)(x)
    return Model(inputs=[inputs1, inputs2], outputs=[x1, x2])


def test__call__if_loss_is_None(model):
    saliency = Saliency(model)
    with pytest.raises(ValueError):
        saliency(None, None)


def test__call__if_seed_input_is_None(model):
    saliency = Saliency(model)
    with pytest.raises(ValueError):
        saliency(CategoricalScore(1), None)


def test__call__if_seed_input_has_not_batch_dim(model):
    saliency = Saliency(model)
    result = saliency(CategoricalScore(1), np.random.sample((8, 8, 3)))
    assert result.shape == (1, 8, 8)


def test__call__(model):
    saliency = Saliency(model)
    result = saliency(CategoricalScore(1), np.random.sample((1, 8, 8, 3)))
    assert result.shape == (1, 8, 8)


def test__call__if_keepdims_is_active(dense_model):
    saliency = Saliency(dense_model)
    result = saliency(CategoricalScore(1), np.random.sample((3, )), keepdims=True)
    assert result.shape == (1, 3)


def test__call__if_smoothing_is_active(model):
    saliency = Saliency(model)
    result = saliency(CategoricalScore(1), np.random.sample((1, 8, 8, 3)), smooth_samples=1)
    assert result.shape == (1, 8, 8)
    result = saliency(CategoricalScore(1), np.random.sample((1, 8, 8, 3)), smooth_samples=2)
    assert result.shape == (1, 8, 8)


def test__call__if_model_has_only_dense_layer(dense_model):
    saliency = Saliency(dense_model)
    result = saliency(CategoricalScore(1), np.random.sample((3, )), keepdims=True)
    assert result.shape == (1, 3)


def test__call__if_model_has_multiple_inputs(multiple_inputs_model):
    saliency = Saliency(multiple_inputs_model)
    result = saliency(
        CategoricalScore(1), [np.random.sample(
            (1, 8, 8, 3)), np.random.sample((1, 10, 10, 3))])
    assert len(result) == 2
    assert result[0].shape == (1, 8, 8)
    assert result[1].shape == (1, 10, 10)


def test__call__when_model_has_multiple_outputs(multiple_outputs_model):
    saliency = Saliency(multiple_outputs_model)
    result = saliency([CategoricalScore(1), lambda x: x], np.random.sample((1, 8, 8, 3)))
    assert result.shape == (1, 8, 8)


def test__call__if_model_has_multiple_io(multiple_io_model):
    saliency = Saliency(multiple_io_model)
    result = saliency(
        [CategoricalScore(1), lambda x: x],
        [np.random.sample(
            (1, 8, 8, 3)), np.random.sample((1, 10, 10, 3))])
    assert len(result) == 2
    assert result[0].shape == (1, 8, 8)
    assert result[1].shape == (1, 10, 10)
