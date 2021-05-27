import numpy as np
import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.models import load_model

from tf_keras_vis.scorecam import Scorecam as Gradcam
from tf_keras_vis.utils.scores import BinaryScore, CategoricalScore
from tf_keras_vis.utils.test import (MockListOfScore, MockScore, MockTupleOfScore, does_not_raise,
                                     dummy_sample, mock_conv_model,
                                     mock_conv_model_with_float32_output, mock_multiple_io_model)

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import set_global_policy


class TestScorecamWithDenseModel():
    def test__call__(self, dense_model):
        gradcam = Gradcam(dense_model)
        with pytest.raises(ValueError):
            result = gradcam(MockScore(), dummy_sample((1, 8, 8, 3)))
            assert result.shape == (1, 8, 8)


class TestScorecam():
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
        ('dense-1', False, pytest.raises(ValueError)),
        (1, False, does_not_raise()),
        (1, True, does_not_raise()),
        ('conv-1', True, does_not_raise()),
        (0, True, pytest.raises(ValueError)),
        ('input-1', True, pytest.raises(ValueError)),
        (MockScore(), True, pytest.raises(ValueError)),
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

    def test__call__if_max_N_is_1(self, conv_model):
        gradcam = Gradcam(conv_model)
        result = gradcam(MockScore(), dummy_sample((1, 8, 8, 3)), max_N=1)
        assert result.shape == (1, 8, 8)

    @pytest.mark.parametrize("score_class,modifier_enabled,clone_enabled,"
                             "batch_size,expectation", [
                                 (BinaryScore, False, False, 0, does_not_raise()),
                                 (BinaryScore, False, False, 1, does_not_raise()),
                                 (BinaryScore, False, False, 5, does_not_raise()),
                                 (BinaryScore, False, True, 5, does_not_raise()),
                                 (BinaryScore, True, False, 5, does_not_raise()),
                                 (BinaryScore, True, True, 5, does_not_raise()),
                                 (CategoricalScore, False, False, 0, does_not_raise()),
                                 (CategoricalScore, False, False, 1, does_not_raise()),
                                 (CategoricalScore, False, False, 5, does_not_raise()),
                                 (CategoricalScore, False, True, 5, does_not_raise()),
                                 (CategoricalScore, True, False, 5, does_not_raise()),
                                 (CategoricalScore, True, True, 5, does_not_raise()),
                             ])
    def test__call__with_categorical_score(self, score_class, modifier_enabled, clone_enabled,
                                           batch_size, expectation, conv_model, conv_sigmoid_model):
        # Release v.0.6.0@dev(May 22 2021):
        #   Add this case to test Scorecam with ScoreClasses.
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

        with expectation:
            gradcam = Gradcam(model,
                              model_modifier=model_modifier if modifier_enabled else None,
                              clone=clone_enabled)
            result = gradcam(score, seed_input=seed_input)
            if modifier_enabled and clone_enabled:
                assert model is not gradcam.model
            else:
                assert model is gradcam.model
            assert result.shape == (max(batch_size, 1), 8, 8)


class TestScorecamWithMultipleInputsModel():
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


class TestScorecamWithMultipleOutputsModel():
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


class TestScorecamWithMultipleIOModel():
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
class TestScorecamWithMixedPrecision():
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
