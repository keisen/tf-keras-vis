import numpy as np
import pytest
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, Dense, GlobalAveragePooling2D, Input)
from tensorflow.keras.models import Model

from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Callback
from tf_keras_vis.utils.regularizers import Norm, TotalVariation2D
from tf_keras_vis.utils.scores import CategoricalScore


class MockCallback(Callback):
    def on_begin(self):
        self.on_begin_was_called = True

    def __call__(self, i, values, grads, losses, model_outpus, **kwargs):
        self.on_call_was_called = True

    def on_end(self):
        self.on_end_was_called = True


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
    activation_maximization = ActivationMaximization(model)
    with pytest.raises(ValueError):
        activation_maximization(None, steps=1)


def test__call__(model):
    activation_maximization = ActivationMaximization(model)
    result = activation_maximization(CategoricalScore(1), steps=1)
    assert result.shape == (1, 8, 8, 3)


def test__call__if_loss_is_list(model):
    activation_maximization = ActivationMaximization(model)
    result = activation_maximization([CategoricalScore(1)], steps=1)
    assert result.shape == (1, 8, 8, 3)


def test__call__with_seed_input(model):
    activation_maximization = ActivationMaximization(model)
    result = activation_maximization(CategoricalScore(1),
                                     seed_input=np.random.sample((8, 8, 3)),
                                     steps=1)
    assert result.shape == (1, 8, 8, 3)


def test__call__with_callback(model):
    activation_maximization = ActivationMaximization(model)
    mock = MockCallback()
    result = activation_maximization(CategoricalScore(1), steps=1, callbacks=mock)
    assert result.shape == (1, 8, 8, 3)
    assert mock.on_begin_was_called
    assert mock.on_call_was_called
    assert mock.on_end_was_called


def test__call__with_gradient_modifier(model):
    activation_maximization = ActivationMaximization(model)
    result = activation_maximization(CategoricalScore(1), steps=1, gradient_modifier=lambda x: x)
    assert result.shape == (1, 8, 8, 3)


def test__call__with_mutiple_inputs_model(multiple_inputs_model):
    activation_maximization = ActivationMaximization(multiple_inputs_model)
    result = activation_maximization(CategoricalScore(1), steps=1, input_modifiers=None)
    assert result[0].shape == (1, 8, 8, 3)
    assert result[1].shape == (1, 10, 10, 3)


def test__call__with_mutiple_outputs_model(multiple_outputs_model):
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    result = activation_maximization(lambda x: x, steps=1, input_modifiers=None)
    assert result.shape == (1, 8, 8, 3)
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    result = activation_maximization([CategoricalScore(1), lambda x: x],
                                     steps=1,
                                     input_modifiers=None)
    assert result.shape == (1, 8, 8, 3)
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    result = activation_maximization([CategoricalScore(1), lambda x: x],
                                     steps=1,
                                     input_modifiers=None,
                                     regularizers=[TotalVariation2D(10.),
                                                   Norm(10.)])
    assert result.shape == (1, 8, 8, 3)


def test__call__with_mutiple_outputs_model_but_losses_is_too_many(multiple_outputs_model):
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    with pytest.raises(ValueError):
        activation_maximization(
            [CategoricalScore(1), CategoricalScore(1),
             CategoricalScore(1)],
            steps=1,
            input_modifiers=None)
