import numpy as np
import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.models import load_model

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import BinaryScore, CategoricalScore
from tf_keras_vis.utils.test import (NO_ERROR, assert_error, dummy_sample, mock_conv_model,
                                     mock_conv_model_with_float32_output, mock_multiple_io_model,
                                     score_with_list, score_with_tuple)

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import set_global_policy


@pytest.fixture(scope='session', params=[Gradcam, GradcamPlusPlus, Scorecam])
def xcam(request):
    return request.param


class TestXcam():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        (CategoricalScore(0), NO_ERROR),
        (score_with_tuple, NO_ERROR),
        (score_with_list, NO_ERROR),
        ([None], ValueError),
        ([CategoricalScore(0)], NO_ERROR),
        ([score_with_tuple], NO_ERROR),
        ([score_with_list], NO_ERROR),
    ])
    def test__call__if_score_is_(self, scores, expected_error, conv_model, xcam):
        cam = xcam(conv_model)
        with assert_error(expected_error):
            result = cam(scores, dummy_sample((1, 8, 8, 3)))
            assert result.shape == (1, 8, 8)

    @pytest.mark.parametrize("seed_input,expected,expected_error", [
        (None, None, ValueError),
        (dummy_sample((8, )), None, ValueError),
        (dummy_sample((8, 8, 3)), (1, 8, 8), NO_ERROR),
        ([dummy_sample((8, 8, 3))], [(1, 8, 8)], NO_ERROR),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8), NO_ERROR),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8)], NO_ERROR),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected, expected_error, conv_model, xcam):
        cam = xcam(conv_model)
        with assert_error(expected_error):
            result = cam(CategoricalScore(0), seed_input)
            if type(expected) is list:
                assert type(result) is list
                expected = expected[0]
                result = result[0]
            assert result.shape == expected

    @pytest.mark.parametrize("penultimate_layer,seek_penultimate_conv_layer,expected_error", [
        (None, True, NO_ERROR),
        (-1, True, NO_ERROR),
        (-1.0, True, ValueError),
        ('dense_1', True, NO_ERROR),
        ('dense_1', False, ValueError),
        (1, False, NO_ERROR),
        (1, True, NO_ERROR),
        ('conv_1', True, NO_ERROR),
        (0, True, ValueError),
        ('input_1', True, ValueError),
        (CategoricalScore(0), True, ValueError),
        (mock_conv_model().layers[-1], False, ValueError),
    ])
    def test__call__if_penultimate_layer_is(self, penultimate_layer, seek_penultimate_conv_layer,
                                            expected_error, conv_model, xcam):
        cam = xcam(conv_model)
        with assert_error(expected_error):
            result = cam(CategoricalScore(0),
                         dummy_sample((1, 8, 8, 3)),
                         penultimate_layer=penultimate_layer,
                         seek_penultimate_conv_layer=seek_penultimate_conv_layer)
            assert result.shape == (1, 8, 8)

    def test__call__if_expand_cam_is_False(self, conv_model, xcam):
        cam = xcam(conv_model)
        result = cam(CategoricalScore(0), dummy_sample((1, 8, 8, 3)), expand_cam=False)
        assert result.shape == (1, 6, 6)

    @pytest.mark.parametrize("score_class", [BinaryScore, CategoricalScore])
    @pytest.mark.parametrize("modifier_enabled", [False, True])
    @pytest.mark.parametrize("clone_enabled", [False, True])
    @pytest.mark.parametrize("batch_size", [0, 1, 5])
    def test__call__with_categorical_score(self, score_class, modifier_enabled, clone_enabled,
                                           batch_size, conv_model, conv_sigmoid_model, xcam):
        # Release v.0.6.0@dev(May 22 2021):
        #   Add this case to test Scorecam with CAM class.
        def model_modifier(model):
            model.layers[-1].activation = tf.keras.activations.linear

        if score_class is BinaryScore:
            model = conv_sigmoid_model
        else:
            model = conv_model

        score_targets = np.random.randint(0, 1, max(batch_size, 1))
        score = score_class(list(score_targets))

        seed_input_shape = (8, 8, 3)
        if batch_size > 0:
            seed_input_shape = (batch_size, ) + seed_input_shape
        seed_input = dummy_sample(seed_input_shape)

        cam = xcam(model,
                   model_modifier=model_modifier if modifier_enabled else None,
                   clone=clone_enabled)
        result = cam(score, seed_input=seed_input)
        if modifier_enabled and clone_enabled:
            assert model is not cam.model
        else:
            assert model is cam.model
        assert result.shape == (max(batch_size, 1), 8, 8)


class TestXcamWithMultipleInputsModel():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        (CategoricalScore(0), NO_ERROR),
        (score_with_tuple, NO_ERROR),
        (score_with_list, NO_ERROR),
        ([None], ValueError),
        ([CategoricalScore(0)], NO_ERROR),
        ([score_with_tuple], NO_ERROR),
        ([score_with_list], NO_ERROR),
    ])
    def test__call__if_score_is_(self, scores, expected_error, multiple_inputs_model, xcam):
        cam = xcam(multiple_inputs_model)
        with assert_error(expected_error):
            result = cam(scores, [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
            assert len(result) == 2
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)

    @pytest.mark.parametrize("seed_input,expected_error", [
        (None, ValueError),
        (dummy_sample((1, 8, 8, 3)), ValueError),
        ([dummy_sample((1, 8, 8, 3))], ValueError),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], NO_ERROR),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected_error, multiple_inputs_model,
                                      xcam):
        cam = xcam(multiple_inputs_model)
        with assert_error(expected_error):
            result = cam(CategoricalScore(0), seed_input)
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)


class TestXcamWithMultipleOutputsModel():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        ([None], ValueError),
        (CategoricalScore(0), ValueError),
        ([CategoricalScore(0)], ValueError),
        ([None, None], ValueError),
        ([CategoricalScore(0), None], ValueError),
        ([None, CategoricalScore(0)], ValueError),
        ([CategoricalScore(0), CategoricalScore(0)], NO_ERROR),
        ([score_with_tuple, score_with_tuple], NO_ERROR),
        ([score_with_list, score_with_list], NO_ERROR),
    ])
    def test__call__if_score_is_(self, scores, expected_error, multiple_outputs_model, xcam):
        cam = xcam(multiple_outputs_model)
        with assert_error(expected_error):
            result = cam(scores, dummy_sample((1, 8, 8, 3)))
            assert result.shape == (1, 8, 8)

    @pytest.mark.parametrize("seed_input,expected,expected_error", [
        (None, None, ValueError),
        (dummy_sample((8, )), None, ValueError),
        (dummy_sample((8, 8, 3)), (1, 8, 8), NO_ERROR),
        ([dummy_sample((8, 8, 3))], [(1, 8, 8)], NO_ERROR),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8), NO_ERROR),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8)], NO_ERROR),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected, expected_error,
                                      multiple_outputs_model, xcam):
        cam = xcam(multiple_outputs_model)
        with assert_error(expected_error):
            result = cam([CategoricalScore(0), CategoricalScore(0)], seed_input)
            if type(expected) is list:
                assert type(result) is list
                expected = expected[0]
                result = result[0]
            assert result.shape == expected


class TestXcamWithMultipleIOModel():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        ([None], ValueError),
        (CategoricalScore(0), ValueError),
        ([CategoricalScore(0)], ValueError),
        ([None, None], ValueError),
        ([CategoricalScore(0), None], ValueError),
        ([None, CategoricalScore(0)], ValueError),
        ([CategoricalScore(0), CategoricalScore(0)], NO_ERROR),
        ([score_with_tuple, score_with_tuple], NO_ERROR),
        ([score_with_list, score_with_list], NO_ERROR),
    ])
    def test__call__if_score_is_(self, scores, expected_error, multiple_io_model, xcam):
        cam = xcam(multiple_io_model)
        with assert_error(expected_error):
            result = cam(scores, [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)

    @pytest.mark.parametrize("seed_input,expected_error", [
        (None, ValueError),
        (dummy_sample((1, 8, 8, 3)), ValueError),
        ([dummy_sample((1, 8, 8, 3))], ValueError),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], NO_ERROR),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected_error, multiple_io_model, xcam):
        cam = xcam(multiple_io_model)
        with assert_error(expected_error):
            result = cam([CategoricalScore(0), CategoricalScore(0)], seed_input)
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)


class TestXcamWithDenseModel():
    def test__call__(self, dense_model, xcam):
        cam = xcam(dense_model)
        with assert_error(ValueError):
            result = cam(CategoricalScore(0), dummy_sample((1, 8, 8, 3)))
            assert result.shape == (1, 8, 8)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled only when tensorflow version is 2.4.0+.")
class TestXcamOnMixedPrecision(TestXcam):
    @classmethod
    def setup_class(cls):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    @classmethod
    def teardown_class(cls):
        tf.keras.mixed_precision.set_global_policy('float32')


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled only when tensorflow version is 2.4.0+.")
class TestXcamWithMultipleInputsModelOnMixedPrecision(TestXcamWithMultipleInputsModel):
    @classmethod
    def setup_class(cls):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    @classmethod
    def teardown_class(cls):
        tf.keras.mixed_precision.set_global_policy('float32')


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled only when tensorflow version is 2.4.0+.")
class TestXcamWithMultipleOutputsModelOnMixedPrecision(TestXcamWithMultipleOutputsModel):
    @classmethod
    def setup_class(cls):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    @classmethod
    def teardown_class(cls):
        tf.keras.mixed_precision.set_global_policy('float32')


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled only when tensorflow version is 2.4.0+.")
class TestXcamWithMultipleIOModelOnMixedPrecision(TestXcamWithMultipleIOModel):
    @classmethod
    def setup_class(cls):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    @classmethod
    def teardown_class(cls):
        tf.keras.mixed_precision.set_global_policy('float32')


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled only when tensorflow version is 2.4.0+.")
class TestXcamWithDenseModelOnMixedPrecision(TestXcamWithDenseModel):
    @classmethod
    def setup_class(cls):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    @classmethod
    def teardown_class(cls):
        tf.keras.mixed_precision.set_global_policy('float32')


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
class TestMixedPrecision():
    def test__call__with_single_io(self, tmpdir, xcam):
        # Create and save lower precision model
        set_global_policy('mixed_float16')
        model = mock_conv_model()
        self._test_for_single_io(model, xcam)
        path = tmpdir.mkdir("tf-keras-vis").join("single_io.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_single_io(model, xcam)
        # Load and test lower precision model on full precision environment
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model, xcam)

    def test__call__with_float32_output_model(self, tmpdir, xcam):
        # Create and save lower precision model
        set_global_policy('mixed_float16')
        model = mock_conv_model_with_float32_output()
        self._test_for_single_io(model, xcam)
        path = tmpdir.mkdir("tf-keras-vis").join("float32_output.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_single_io(model, xcam)
        # Load and test lower precision model on full precision environment
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model, xcam)

    def _test_for_single_io(self, model, xcam):
        cam = xcam(model)
        result = cam(CategoricalScore(0), dummy_sample((1, 8, 8, 3)))
        assert result.shape == (1, 8, 8)

    def test__call__with_multiple_io(self, tmpdir, xcam):
        # Create and save lower precision model
        set_global_policy('mixed_float16')
        model = mock_multiple_io_model()
        self._test_for_multiple_io(model, xcam)
        path = tmpdir.mkdir("tf-keras-vis").join("multiple_io.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_multiple_io(model, xcam)
        # Load and test lower precision model on full precision environment
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_multiple_io(model, xcam)

    def _test_for_multiple_io(self, model, xcam):
        cam = xcam(model)
        result = cam([CategoricalScore(0), CategoricalScore(0)],
                     [dummy_sample(
                         (1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
        assert result[0].shape == (1, 8, 8)
        assert result[1].shape == (1, 10, 10)
