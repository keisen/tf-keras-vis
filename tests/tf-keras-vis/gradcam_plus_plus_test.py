import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.models import load_model

from tf_keras_vis.gradcam import GradcamPlusPlus as Gradcam
from tf_keras_vis.utils.test import (MockListOfScore, MockScore, MockTupleOfScore, does_not_raise,
                                     dummy_sample, mock_conv_model,
                                     mock_conv_model_with_float32_output, mock_multiple_io_model)

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import set_global_policy


class TestGradcamPlusPlusWithDenseModel():
    def test__call__(self, dense_model):
        gradcam = Gradcam(dense_model)
        with pytest.raises(ValueError):
            result = gradcam(MockScore(), dummy_sample((1, 8, 8, 3)))
            assert result.shape == (1, 8, 8)


class TestGradcamPlusPlus():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        (MockTupleOfScore(), does_not_raise()),
        (MockListOfScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, conv_model):
        gradcam = Gradcam(conv_model)
        with expectation:
            result = gradcam(scores, dummy_sample((1, 8, 8, 3)))
            assert result.shape == (1, 8, 8)

    @pytest.mark.parametrize("seed_input,expected,expectation", [
        (None, None, pytest.raises(ValueError)),
        (dummy_sample((8, )), None, pytest.raises(ValueError)),
        (dummy_sample((8, 8, 3)), (1, 8, 8), does_not_raise()),
        ([dummy_sample((8, 8, 3))], [(1, 8, 8)], does_not_raise()),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8), does_not_raise()),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8)], does_not_raise()),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected, expectation, conv_model):
        gradcam = Gradcam(conv_model)
        with expectation:
            result = gradcam(MockScore(), seed_input)
            if type(expected) is list:
                assert type(result) is list
                expected = expected[0]
                result = result[0]
            assert result.shape == expected

    @pytest.mark.parametrize("penultimate_layer,seek_penultimate_conv_layer,expectation", [
        (None, True, does_not_raise()),
        (-1, True, does_not_raise()),
        ('dense-1', True, does_not_raise()),
        (1, False, does_not_raise()),
        (1, True, does_not_raise()),
        ('conv-1', True, does_not_raise()),
        (0, True, pytest.raises(ValueError)),
        ('input-1', True, pytest.raises(ValueError)),
        (MockScore(), True, pytest.raises(ValueError)),
        (mock_conv_model().layers[-1], False, pytest.raises(ValueError)),
    ])
    def test__call__if_penultimate_layer_is(self, penultimate_layer, seek_penultimate_conv_layer,
                                            expectation, conv_model):
        gradcam = Gradcam(conv_model)
        with expectation:
            result = gradcam(MockScore(),
                             dummy_sample((1, 8, 8, 3)),
                             penultimate_layer=penultimate_layer,
                             seek_penultimate_conv_layer=seek_penultimate_conv_layer)
            assert result.shape == (1, 8, 8)

    def test__call__if_expand_cam_is_False(self, conv_model):
        gradcam = Gradcam(conv_model)
        result = gradcam(MockScore(), dummy_sample((1, 8, 8, 3)), expand_cam=False)
        assert result.shape == (1, 6, 6)

    def test__call__if_activation_modifier_is_None(self, conv_model):
        gradcam = Gradcam(conv_model)
        result = gradcam(MockScore(), dummy_sample((1, 8, 8, 3)), activation_modifier=None)
        assert result.shape == (1, 8, 8)


class TestGradcamPlusPlusWithMultipleInputsModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        (MockTupleOfScore(), does_not_raise()),
        (MockListOfScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_inputs_model):
        gradcam = Gradcam(multiple_inputs_model)
        with expectation:
            result = gradcam(scores, [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
            assert len(result) == 2
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)

    @pytest.mark.parametrize("seed_input,expectation", [
        (None, pytest.raises(ValueError)),
        (dummy_sample((1, 8, 8, 3)), pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3))], pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], does_not_raise()),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expectation, multiple_inputs_model):
        gradcam = Gradcam(multiple_inputs_model)
        with expectation:
            result = gradcam(MockScore(), seed_input)
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)


class TestGradcamPlusPlusWithMultipleOutputsModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        ([None], pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
        ([None, None], pytest.raises(ValueError)),
        ([MockScore(), None], pytest.raises(ValueError)),
        ([MockScore(), MockScore()], does_not_raise()),
        ([MockTupleOfScore(), MockTupleOfScore()], does_not_raise()),
        ([MockListOfScore(), MockListOfScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_outputs_model):
        gradcam = Gradcam(multiple_outputs_model)
        with expectation:
            result = gradcam(scores, dummy_sample((1, 8, 8, 3)))
            assert result.shape == (1, 8, 8)

    @pytest.mark.parametrize("seed_input,expected,expectation", [
        (None, None, pytest.raises(ValueError)),
        (dummy_sample((8, )), None, pytest.raises(ValueError)),
        (dummy_sample((8, 8, 3)), (1, 8, 8), does_not_raise()),
        ([dummy_sample((8, 8, 3))], [(1, 8, 8)], does_not_raise()),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8), does_not_raise()),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8)], does_not_raise()),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected, expectation,
                                      multiple_outputs_model):
        gradcam = Gradcam(multiple_outputs_model)
        with expectation:
            result = gradcam(MockScore(), seed_input)
            if type(expected) is list:
                assert type(result) is list
                expected = expected[0]
                result = result[0]
            assert result.shape == expected


class TestGradcamPlusPlusWithMultipleIOModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        ([None], pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
        ([None, None], pytest.raises(ValueError)),
        ([MockScore(), None], pytest.raises(ValueError)),
        ([MockScore(), MockScore()], does_not_raise()),
        ([MockTupleOfScore(), MockTupleOfScore()], does_not_raise()),
        ([MockListOfScore(), MockListOfScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_io_model):
        gradcam = Gradcam(multiple_io_model)
        with expectation:
            result = gradcam(scores, [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)

    @pytest.mark.parametrize("seed_input,expectation", [
        (None, pytest.raises(ValueError)),
        (dummy_sample((1, 8, 8, 3)), pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3))], pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], does_not_raise()),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expectation, multiple_io_model):
        gradcam = Gradcam(multiple_io_model)
        with expectation:
            result = gradcam(MockScore(), seed_input)
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
class TestGradcamPlusPlusWithMixedPrecision():
    def test__call__with_single_io(self, tmpdir):
        set_global_policy('mixed_float16')
        model = mock_conv_model()
        self._test_for_single_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("single_io.h5")
        model.save(path)
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model)

    def test__call__with_float32_output_model(self, tmpdir):
        set_global_policy('mixed_float16')
        model = mock_conv_model_with_float32_output()
        self._test_for_single_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("float32_output.h5")
        model.save(path)
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model)

    def _test_for_single_io(self, model):
        gradcam = Gradcam(model)
        result = gradcam(MockScore(), dummy_sample((1, 8, 8, 3)))
        assert result.shape == (1, 8, 8)

    def test__call__with_multiple_io(self, tmpdir):
        set_global_policy('mixed_float16')
        model = mock_multiple_io_model()
        self._test_for_multiple_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("multiple_io.h5")
        model.save(path)
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_multiple_io(model)

    def _test_for_multiple_io(self, model):
        gradcam = Gradcam(model)
        result = gradcam(MockScore(), [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
        assert result[0].shape == (1, 8, 8)
        assert result[1].shape == (1, 10, 10)
