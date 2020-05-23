import numpy as np
import pytest
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential

from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import OptimizerCallback
from tf_keras_vis.utils.losses import SmoothedLoss
from tf_keras_vis.utils.regularizers import L2Norm, TotalVariation


class MockCallback(OptimizerCallback):
    def on_begin(self):
        self.on_begin_was_called = True

    def __call__(self, i, values, grads, losses, model_outpus, **kwargs):
        self.on_call_was_called = True

    def on_end(self):
        self.on_end_was_called = True


@pytest.fixture(scope="function", autouse=True)
def multiple_inputs_model():
    a = Input(shape=(8, 8, 3))
    b = Input(shape=(8, 8, 3))
    c = Input(shape=(8, 8, 3))
    x1 = Conv2D(5, 3, activation='relu')(a)
    x2 = Conv2D(5, 3, activation='relu')(b)
    x3 = Conv2D(5, 3, activation='relu')(c)
    x = Concatenate()([x1, x2, x3])
    x = Dense(3)(x)
    return Model([a, b, c], [x])


@pytest.fixture(scope="function", autouse=True)
def multiple_outputs_model():
    x = i = Input(shape=(8, 8, 3))
    x = Conv2D(5, 3, activation='relu')(x)
    x = Flatten()(x)
    a = Dense(3, name='output_a')(x)
    b = Dense(2, name='output_b')(x)
    return Model([i], [a, b])


@pytest.fixture(scope="function", autouse=True)
def cnn_model():
    return _cnn_model()


def _cnn_model():
    return Sequential([
        Input(shape=(8, 8, 3)),
        Conv2D(5, 3, activation='relu'),
        Flatten(),
        Dense(2, activation='softmax')
    ])


def test__call__if_loss_is_None(cnn_model):
    activation_maximization = ActivationMaximization(cnn_model)
    try:
        activation_maximization(None, steps=1)
        assert False
    except ValueError:
        assert True


def test__call__(cnn_model):
    activation_maximization = ActivationMaximization(cnn_model)
    result = activation_maximization(SmoothedLoss(1), steps=1)
    assert result.shape == (1, 8, 8, 3)


def test__call__if_loss_is_list(cnn_model):
    activation_maximization = ActivationMaximization(cnn_model)
    result = activation_maximization([SmoothedLoss(1)], steps=1)
    assert result.shape == (1, 8, 8, 3)


def test__call__with_seed_input(cnn_model):
    activation_maximization = ActivationMaximization(cnn_model)
    result = activation_maximization(SmoothedLoss(1),
                                     seed_input=np.random.sample((8, 8, 3)),
                                     steps=1)
    assert result.shape == (1, 8, 8, 3)


def test__call__with_callback(cnn_model):
    activation_maximization = ActivationMaximization(cnn_model)
    mock = MockCallback()
    result = activation_maximization(SmoothedLoss(1), steps=1, callbacks=mock)
    assert result.shape == (1, 8, 8, 3)
    assert mock.on_begin_was_called
    assert mock.on_call_was_called
    assert mock.on_end_was_called


def test__call__with_gradient_modifier(cnn_model):
    activation_maximization = ActivationMaximization(cnn_model)
    result = activation_maximization(SmoothedLoss(1), steps=1, gradient_modifier=lambda x: x)
    assert result.shape == (1, 8, 8, 3)


def test__call__with_mutiple_inputs_model(multiple_inputs_model):
    activation_maximization = ActivationMaximization(multiple_inputs_model)
    result = activation_maximization(SmoothedLoss(1), steps=1, input_modifiers=None)
    assert result[0].shape == (1, 8, 8, 3)
    assert result[1].shape == (1, 8, 8, 3)


def test__call__with_mutiple_outputs_model(multiple_outputs_model):
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    result = activation_maximization(SmoothedLoss(1), steps=1, input_modifiers=None)
    assert result.shape == (1, 8, 8, 3)
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    result = activation_maximization([SmoothedLoss(1), SmoothedLoss(1)],
                                     steps=1,
                                     input_modifiers=None)
    assert result.shape == (1, 8, 8, 3)
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    result = activation_maximization([SmoothedLoss(1), SmoothedLoss(1)],
                                     steps=1,
                                     input_modifiers=None,
                                     regularizers=[TotalVariation(10.),
                                                   L2Norm(10.)])
    assert result.shape == (1, 8, 8, 3)


def test__call__with_mutiple_outputs_model_but_losses_is_too_many(multiple_outputs_model):
    activation_maximization = ActivationMaximization(multiple_outputs_model)
    try:
        activation_maximization(
            [SmoothedLoss(1), SmoothedLoss(1), SmoothedLoss(1)], steps=1, input_modifiers=None)
        assert False
    except ValueError:
        assert True
