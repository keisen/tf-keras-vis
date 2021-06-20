import pytest
import tensorflow as tf
from packaging.version import parse as version

from tf_keras_vis.utils.test import (mock_conv_model, mock_conv_model_with_sigmoid_output,
                                     mock_dense_model, mock_multiple_inputs_model,
                                     mock_multiple_io_model, mock_multiple_outputs_model)


@pytest.fixture(scope='function',
                params=["float32"]
                if version(tf.version.VERSION) < version("2.4.0") else ["float32", "mixed_float16"])
def mixed_precision(request):
    if version(tf.version.VERSION) >= version("2.4.0"):
        tf.keras.mixed_precision.set_global_policy(request.param)
    yield
    if version(tf.version.VERSION) >= version("2.4.0"):
        tf.keras.mixed_precision.set_global_policy("float32")


@pytest.fixture
def dense_model():
    return mock_dense_model()


@pytest.fixture
def conv_model():
    return mock_conv_model()


@pytest.fixture
def conv_sigmoid_model():
    return mock_conv_model_with_sigmoid_output()


@pytest.fixture
def multiple_inputs_model():
    return mock_multiple_inputs_model()


@pytest.fixture
def multiple_outputs_model():
    return mock_multiple_outputs_model()


@pytest.fixture
def multiple_io_model():
    return mock_multiple_io_model()
