import pytest

from tf_keras_vis.utils.test import (mock_conv_model, mock_conv_model_with_sigmoid_output,
                                     mock_dense_model, mock_multiple_inputs_model,
                                     mock_multiple_io_model, mock_multiple_outputs_model)


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
