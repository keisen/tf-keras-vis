import os

import pytest
import tensorflow as tf
from packaging.version import parse as version

from tf_keras_vis.utils.test import (mock_conv_model, mock_conv_model_with_sigmoid_output,
                                     mock_dense_model, mock_multiple_inputs_model,
                                     mock_multiple_io_model, mock_multiple_outputs_model)


def _get_supported_policies():
    if version(tf.version.VERSION) < version("2.4.0"):
        return ["float32"]
    else:
        return ["float32", "mixed_float16"]


def _source_of_models():
    return [None] + _get_supported_policies()


def _save_and_load(model, source, path):
    if source is None:
        return model
    if source == "mixed_float16":
        policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy(source)
        try:
            model.save(path)
        finally:
            tf.keras.mixed_precision.set_global_policy(policy)
    else:
        model.save(path)
    return tf.keras.models.load_model(path)


@pytest.fixture(scope='function', params=_get_supported_policies())
def mixed_precision(request):
    if version(tf.version.VERSION) >= version("2.4.0"):
        tf.keras.mixed_precision.set_global_policy(request.param)
    yield
    if version(tf.version.VERSION) >= version("2.4.0"):
        tf.keras.mixed_precision.set_global_policy("float32")


@pytest.fixture(scope='function', params=_source_of_models())
def dense_model(request, tmpdir):
    return _save_and_load(mock_dense_model(), request.param,
                          os.path.join(tmpdir, 'dense-model.h5'))


@pytest.fixture(scope='function', params=_source_of_models())
def conv_model(request, tmpdir):
    return _save_and_load(mock_conv_model(), request.param, os.path.join(tmpdir, 'conv-model.h5'))


@pytest.fixture(scope='function', params=_source_of_models())
def conv_sigmoid_model(request, tmpdir):
    return _save_and_load(mock_conv_model_with_sigmoid_output(), request.param,
                          os.path.join(tmpdir, 'conv-model-with-sigmoid-output.h5'))


@pytest.fixture(scope='function', params=_source_of_models())
def multiple_inputs_model(request, tmpdir):
    return _save_and_load(mock_multiple_inputs_model(), request.param,
                          os.path.join(tmpdir, 'multiple-inputs-model.h5'))


@pytest.fixture(scope='function', params=_source_of_models())
def multiple_outputs_model(request, tmpdir):
    return _save_and_load(mock_multiple_outputs_model(), request.param,
                          os.path.join(tmpdir, 'multiple-outputs-model.h5'))


@pytest.fixture(scope='function', params=_source_of_models())
def multiple_io_model(request, tmpdir):
    return _save_and_load(mock_multiple_io_model(), request.param,
                          os.path.join(tmpdir, 'multiple-io-model.h5'))
