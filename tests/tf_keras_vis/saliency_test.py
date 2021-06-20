import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.models import load_model

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import BinaryScore, CategoricalScore
from tf_keras_vis.utils.test import (dummy_sample, mock_conv_model,
                                     mock_conv_model_with_float32_output, mock_multiple_io_model)

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import set_global_policy


class TestSaliency():
    @pytest.mark.parametrize("keepdims,expected", [
        (False, (1, 8, 8)),
        (True, (1, 8, 8, 3)),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_keepdims_is_(self, keepdims, expected, conv_model):
        saliency = Saliency(conv_model)
        result = saliency(CategoricalScore(0), dummy_sample((1, 8, 8, 3)), keepdims=keepdims)
        assert result.shape == expected

    @pytest.mark.parametrize("smooth_samples", [1, 3, 100])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_smoothing_is_active(self, smooth_samples, conv_model):
        saliency = Saliency(conv_model)
        result = saliency(CategoricalScore(0),
                          dummy_sample((1, 8, 8, 3)),
                          smooth_samples=smooth_samples)
        assert result.shape == (1, 8, 8)

    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_model_has_only_dense_layers(self, dense_model):
        saliency = Saliency(dense_model)
        result = saliency(CategoricalScore(0), dummy_sample((8, )), keepdims=True)
        assert result.shape == (1, 8)


class TestSaliencyWithMultipleInputsModel():
    @pytest.mark.parametrize("keepdims,expected", [
        (False, [(1, 8, 8), (1, 10, 10)]),
        (True, [(1, 8, 8, 3), (1, 10, 10, 3)]),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_keepdims_is_(self, keepdims, expected, multiple_inputs_model):
        saliency = Saliency(multiple_inputs_model)
        result = saliency(
            CategoricalScore(0), [dummy_sample(
                (1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))],
            keepdims=keepdims)
        assert len(result) == 2
        assert result[0].shape == expected[0]
        assert result[1].shape == expected[1]

    @pytest.mark.parametrize("smooth_samples", [1, 3, 100])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__if_smoothing_is_active(self, smooth_samples, multiple_inputs_model):
        saliency = Saliency(multiple_inputs_model)
        result = saliency(
            CategoricalScore(0), [dummy_sample(
                (1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))],
            smooth_samples=smooth_samples)
        assert len(result) == 2
        assert result[0].shape == (1, 8, 8)
        assert result[1].shape == (1, 10, 10)


class TestSaliencyWithMultipleOutputsModel():
    pass


class TestSaliencyWithMultipleIOModel():
    pass


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
class TestMixedPrecision():
    def test__call__with_single_io(self, tmpdir):
        # Create and save lower precision model
        set_global_policy('mixed_float16')
        model = mock_conv_model()
        self._test_for_single_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("single_io.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_single_io(model)
        # Load and test lower precision model on full precision environment
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model)

    def test__call__with_float32_output_model(self, tmpdir):
        # Create and save lower precision model
        set_global_policy('mixed_float16')
        model = mock_conv_model_with_float32_output()
        self._test_for_single_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("float32_output.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_single_io(model)
        # Load and test lower precision model on full precision environment
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model)

    def _test_for_single_io(self, model):
        saliency = Saliency(model)
        result = saliency(CategoricalScore(0), dummy_sample((1, 8, 8, 3)))
        assert result.shape == (1, 8, 8)

    def test__call__with_multiple_io(self, tmpdir):
        # Create and save lower precision model
        set_global_policy('mixed_float16')
        model = mock_multiple_io_model()
        self._test_for_multiple_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("multiple_io.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_multiple_io(model)
        # Load and test lower precision model on full precision environment
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_multiple_io(model)

    def _test_for_multiple_io(self, model):
        saliency = Saliency(model)
        result = saliency(
            [CategoricalScore(0), BinaryScore(0)],
            [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
        assert len(result) == 2
        assert result[0].shape == (1, 8, 8)
        assert result[1].shape == (1, 10, 10)
