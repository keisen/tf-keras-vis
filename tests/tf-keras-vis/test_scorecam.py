import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, Dense, GlobalAveragePooling2D, Input)
from tensorflow.keras.models import Model

from tf_keras_vis.scorecam import ScoreCAM
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
    x = Conv2D(5, 3, activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)


@pytest.fixture(scope="function", autouse=True)
def multiple_inputs_model():
    inputs1 = Input((8, 8, 3))
    inputs2 = Input((10, 10, 3))
    x1 = Conv2D(5, 3, padding='same', activation='relu')(inputs1)
    x2 = Conv2D(5, 3, activation='relu')(inputs2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=[inputs1, inputs2], outputs=x)


@pytest.fixture(scope="function", autouse=True)
def multiple_outputs_model():
    inputs = Input((8, 8, 3))
    x = Conv2D(5, 3, activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, activation='softmax')(x)
    x2 = Dense(1)(x)
    return Model(inputs=inputs, outputs=[x1, x2])


@pytest.fixture(scope="function", autouse=True)
def multiple_io_model():
    inputs1 = Input((8, 8, 3))
    inputs2 = Input((10, 10, 3))
    x1 = Conv2D(5, 3, padding='same', activation='relu')(inputs1)
    x2 = Conv2D(5, 3, activation='relu')(inputs2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, activation='softmax')(x)
    x2 = Dense(1)(x)
    return Model(inputs=[inputs1, inputs2], outputs=[x1, x2])


def test__call__if_loss_is_None(model):
    scorecam = ScoreCAM(model)
    with pytest.raises(ValueError):
        scorecam(None, None, max_N=3)


def test__call__if_seed_input_is_None(model):
    scorecam = ScoreCAM(model)
    with pytest.raises(ValueError):
        scorecam(CategoricalScore(1), None, max_N=3)


def test__call__if_seed_input_shape_is_invalid(model):
    scorecam = ScoreCAM(model)
    try:
        scorecam(CategoricalScore(1), np.random.sample((8, )))
        assert False
    except (ValueError, tf.errors.InvalidArgumentError):
        # TF became to raise InvalidArgumentError from ver.2.0.2.
        assert True


def test__call__if_seed_input_has_not_batch_dim(model):
    scorecam = ScoreCAM(model)
    result = scorecam(CategoricalScore(1), np.random.sample((8, 8, 3)), max_N=3)
    assert result.shape == (1, 8, 8)


def test__call__(model):
    scorecam = ScoreCAM(model)
    result = scorecam(CategoricalScore(1), np.random.sample((1, 8, 8, 3)), max_N=3)
    assert result.shape == (1, 8, 8)


def test__call__if_penultimate_layer_is_None(model):
    scorecam = ScoreCAM(model)
    result = scorecam(CategoricalScore(1),
                      np.random.sample((1, 8, 8, 3)),
                      penultimate_layer=None,
                      max_N=3)
    assert result.shape == (1, 8, 8)


def test__call__if_penultimate_layer_is_no_exist_index(model):
    scorecam = ScoreCAM(model)
    with pytest.raises(ValueError):
        scorecam(CategoricalScore(1),
                 np.random.sample((1, 8, 8, 3)),
                 penultimate_layer=100000,
                 max_N=3)


def test__call__if_penultimate_layer_is_no_exist_name(model):
    scorecam = ScoreCAM(model)
    with pytest.raises(ValueError):
        scorecam(CategoricalScore(1),
                 np.random.sample((1, 8, 8, 3)),
                 penultimate_layer='hoge',
                 max_N=3)


def test__call__if_model_has_only_dense_layer(dense_model):
    scorecam = ScoreCAM(dense_model)
    with pytest.raises(ValueError):
        scorecam(CategoricalScore(1), np.random.sample((1, 3)))


def test__call__if_model_has_multiple_inputs(multiple_inputs_model):
    scorecam = ScoreCAM(multiple_inputs_model)
    result = scorecam(
        CategoricalScore(1), [np.random.sample(
            (1, 8, 8, 3)), np.random.sample((1, 10, 10, 3))],
        max_N=3)
    assert len(result) == 2
    assert result[0].shape == (1, 8, 8)
    assert result[1].shape == (1, 10, 10)


def test__call__if_model_has_multiple_outputs(multiple_outputs_model):
    scorecam = ScoreCAM(multiple_outputs_model)
    result = scorecam([CategoricalScore(1), lambda x: x], np.random.sample((1, 8, 8, 3)), max_N=3)
    assert result.shape == (1, 8, 8)


def test__call__if_model_has_multiple_io(multiple_io_model):
    scorecam = ScoreCAM(multiple_io_model)
    result = scorecam(
        [CategoricalScore(1), lambda x: x],
        [np.random.sample(
            (1, 8, 8, 3)), np.random.sample((1, 10, 10, 3))],
        max_N=3)
    assert len(result) == 2
    assert result[0].shape == (1, 8, 8)
    assert result[1].shape == (1, 10, 10)


def test__call__if_model_has_multiple_io_when_batchsize_is_2(multiple_io_model):
    scorecam = ScoreCAM(multiple_io_model)
    result = scorecam(
        [CategoricalScore([1, 0]), lambda x: x],
        [np.random.sample(
            (2, 8, 8, 3)), np.random.sample((2, 10, 10, 3))],
        max_N=3)
    assert len(result) == 2
    assert result[0].shape == (2, 8, 8)
    assert result[1].shape == (2, 10, 10)


def test__call__if_expand_cam_is_False(model):
    scorecam = ScoreCAM(model)
    result = scorecam(CategoricalScore(1),
                      np.random.sample((1, 8, 8, 3)),
                      expand_cam=False,
                      max_N=3)
    assert result.shape == (1, 6, 6)


def test__call__if_expand_cam_is_False_and_model_has_multiple_inputs(multiple_inputs_model):
    scorecam = ScoreCAM(multiple_inputs_model)
    result = scorecam(
        CategoricalScore(1), [np.random.sample(
            (1, 8, 8, 3)), np.random.sample((1, 10, 10, 3))],
        expand_cam=False,
        max_N=3)
    assert result.shape == (1, 8, 8)
