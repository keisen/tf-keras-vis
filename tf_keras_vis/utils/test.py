from contextlib import contextmanager

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

from ..activation_maximization.callbacks import Callback


def mock_dense_model():
    inputs = Input((8, ), name='input_1')
    x = Dense(6, activation='relu', name='dense_1')(inputs)
    x = Dense(2, name='dense_2')(x)
    x = Activation('softmax', dtype=tf.float32, name='output_1')(x)
    return Model(inputs=inputs, outputs=x)


def mock_conv_model_with_sigmoid_output():
    inputs = Input((8, 8, 3), name='input_1')
    x = Conv2D(6, 3, activation='relu', name='conv_1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, name='dense_1')(x)
    x = Activation('sigmoid', dtype=tf.float32, name='output_1')(x)
    return Model(inputs=inputs, outputs=x)


def mock_conv_model():
    inputs = Input((8, 8, 3), name='input_1')
    x = Conv2D(6, 3, activation='relu', name='conv_1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, name='dense_1')(x)
    x = Activation('softmax', dtype=tf.float32, name='output_1')(x)
    return Model(inputs=inputs, outputs=x)


def mock_multiple_inputs_model():
    input_1 = Input((8, 8, 3), name='input_1')
    input_2 = Input((10, 10, 3), name='input_2')
    x1 = Conv2D(6, 3, padding='same', activation='relu', name='conv_1')(input_1)
    x2 = Conv2D(6, 3, activation='relu', name='conv_2')(input_2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, name='dense_1')(x)
    x = Activation('softmax', dtype=tf.float32, name='output_1')(x)
    return Model(inputs=[input_1, input_2], outputs=x)


def mock_multiple_outputs_model():
    inputs = Input((8, 8, 3), name='input_1')
    x = Conv2D(6, 3, activation='relu', name='conv_1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, name='dense_1')(x)
    x2 = Dense(1, name='dense_2')(x)
    x1 = Activation('softmax', dtype=tf.float32, name='output_1')(x1)
    x2 = Activation('sigmoid', dtype=tf.float32, name='output_2')(x2)
    return Model(inputs=inputs, outputs=[x1, x2])


def mock_multiple_io_model():
    input_1 = Input((8, 8, 3), name='input_1')
    input_2 = Input((10, 10, 3), name='input_2')
    x1 = Conv2D(6, 3, padding='same', activation='relu', name='conv_1')(input_1)
    x2 = Conv2D(6, 3, activation='relu', name='conv_2')(input_2)
    x = K.concatenate([x1, x2], axis=-1)
    x = GlobalAveragePooling2D()(x)
    x1 = Dense(2, name='dense_1')(x)
    x2 = Dense(1, name='dense_2')(x)
    x1 = Activation('softmax', dtype=tf.float32, name='output_1')(x1)
    x2 = Activation('sigmoid', dtype=tf.float32, name='output_2')(x2)
    return Model(inputs=[input_1, input_2], outputs=[x1, x2])


def mock_conv_model_with_float32_output():
    inputs = Input((8, 8, 3), name='input_1')
    x = Conv2D(6, 3, activation='relu', name='conv_1')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, dtype=tf.float32, activation='softmax', name='dense_1')(x)
    return Model(inputs=inputs, outputs=x)


def dummy_sample(shape, dtype=np.float32):
    length = np.prod(shape)
    values = np.array(list(range(length)))
    values = np.reshape(values, shape)
    values = values.astype(dtype)
    return values


def score_with_tensor(output):
    return output[:, 0]


def score_with_tuple(output):
    return tuple(o[0] for o in output)


def score_with_list(output):
    return list(o[0] for o in output)


NO_ERROR = None


@contextmanager
def _does_not_raise():
    yield


def assert_raises(e):
    if e is NO_ERROR:
        return _does_not_raise()
    else:
        return pytest.raises(e)


class MockCallback(Callback):
    def __init__(self,
                 raise_error_on_begin=False,
                 raise_error_on_call=False,
                 raise_error_on_end=False):
        self.on_begin_was_called = False
        self.on_call_was_called = False
        self.on_end_was_called = False
        self.raise_error_on_begin = raise_error_on_begin
        self.raise_error_on_call = raise_error_on_call
        self.raise_error_on_end = raise_error_on_end

    def on_begin(self, **kwargs):
        self.on_begin_was_called = True
        self.kwargs = kwargs
        if self.raise_error_on_begin:
            raise ValueError('Test')

    def __call__(self, *args, **kwargs):
        self.on_call_was_called = True
        self.args = args
        self.kwargs = kwargs
        if self.raise_error_on_call:
            raise ValueError('Test')

    def on_end(self):
        self.on_end_was_called = True
        if self.raise_error_on_end:
            raise ValueError('Test')


class MockLegacyCallback(Callback):
    def __init__(self, callback):
        self.callback = callback

    def on_begin(self):
        self.callback.on_begin()

    def __call__(self, *args, **kwargs):
        self.callback(*args, **kwargs)

    def on_end(self):
        self.callback.on_end()
