from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, Dense, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.models import Model

from tf_keras_vis.activation_maximization.callbacks import Callback
from tf_keras_vis.utils.input_modifiers import InputModifier
from tf_keras_vis.utils.regularizers import Regularizer
from tf_keras_vis.utils.scores import Score


def mock_dense_model():
    inputs = Input((3, ), name='input-1')
    x = Dense(5, activation='relu', name='dense-1')(inputs)
    x = Dense(2, activation='softmax', name='dense-2')(x)
    return Model(inputs=inputs, outputs=x)


def mock_conv_model_with_sigmoid_output():
    inputs = Input((8, 8, 3), name='input-1')
    x = Conv2D(6, 3, activation='relu', name='conv-1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid', name='dense-1')(x)
    return Model(inputs=inputs, outputs=x)


def mock_conv_model():
    inputs = Input((8, 8, 3), name='input-1')
    x = Conv2D(6, 3, activation='relu', name='conv-1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax', name='dense-1')(x)
    return Model(inputs=inputs, outputs=x)


def mock_multiple_inputs_model():
    input_1 = Input((8, 8, 3), name='input-1')
    input_2 = Input((10, 10, 3), name='input-2')
    x1 = Conv2D(6, 3, padding='same', activation='relu', name='conv-1')(input_1)
    x2 = Conv2D(6, 3, activation='relu', name='conv-2')(input_2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax', name='dense-1')(x)
    return Model(inputs=[input_1, input_2], outputs=x)


def mock_multiple_outputs_model():
    inputs = Input((8, 8, 3), name='input-1')
    x = Conv2D(6, 3, activation='relu', name='conv-1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, activation='softmax', name='dense-1')(x)
    x2 = Dense(1, name='dense-2')(x)
    return Model(inputs=inputs, outputs=[x1, x2])


def mock_multiple_io_model():
    input_1 = Input((8, 8, 3), name='input-1')
    input_2 = Input((10, 10, 3), name='input-2')
    x1 = Conv2D(6, 3, padding='same', activation='relu', name='conv-1')(input_1)
    x2 = Conv2D(6, 3, activation='relu', name='conv-2')(input_2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, activation='softmax', name='dense-1')(x)
    x2 = Dense(1, name='dense-2')(x)
    return Model(inputs=[input_1, input_2], outputs=[x1, x2])


def mock_conv_model_with_flot32_output():
    inputs = Input((8, 8, 3), name='input-1')
    x = Conv2D(6, 3, activation='relu', name='conv-1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, dtype=tf.float32, activation='softmax', name='dense-1')(x)
    return Model(inputs=inputs, outputs=x)


def dummy_sample(shape, dtype=np.float32):
    length = np.prod(shape)
    values = np.array(list(range(length)))
    values = np.reshape(values, shape)
    values = values.astype(dtype)
    return values


@contextmanager
def does_not_raise():
    yield


class MockCallback(Callback):
    def __init__(self):
        self.on_begin_was_called = False
        self.on_call_was_called = False
        self.on_end_was_called = False

    def on_begin(self):
        self.on_begin_was_called = True

    def __call__(self, i, values, grads, losses, model_outpus, **kwargs):
        self.on_call_was_called = True

    def on_end(self):
        self.on_end_was_called = True


class MockInputModifier(InputModifier):
    def __init__(self):
        self.seed_input = None

    def __call__(self, seed_input):
        self.seed_input = seed_input
        return seed_input


class MockScore(Score):
    def __init__(self, name='noname'):
        self.name = name
        self.output = None

    def __call__(self, output):
        self.output = output
        return output


class MockTupleOfScore(Score):
    def __init__(self, name='noname'):
        self.name = name
        self.output = None

    def __call__(self, output):
        self.output = output
        return tuple(o for o in output)


class MockListOfScore(Score):
    def __init__(self, name='noname'):
        self.name = name
        self.output = None

    def __call__(self, output):
        self.output = output
        return list(o for o in output)


class MockRegularizer(Regularizer):
    def __init__(self, name='noname'):
        self.name = name
        self.inputs = None

    def __call__(self, inputs):
        self.inputs = inputs
        return inputs


class MockGradientModifier():
    def __init__(self):
        self.gradients = None

    def __call__(self, gradients):
        self.gradients = gradients
        return gradients
