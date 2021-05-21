from tf_keras_vis.utils.scores import CategoricalScore
import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.models import load_model

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.test import (MockListOfScore, MockScore, MockTupleOfScore, does_not_raise,
                                     dummy_sample, mock_conv_model,
                                     mock_conv_model_with_flot32_output, mock_multiple_io_model)

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import set_global_policy


class TestSaliency():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        (MockTupleOfScore(), does_not_raise()),
        (MockListOfScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, conv_model):
        saliency = Saliency(conv_model)
        with expectation:
            result = saliency(scores, dummy_sample((1, 8, 8, 3)))
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
        saliency = Saliency(conv_model)
        with expectation:
            result = saliency(MockScore(), seed_input)
            if type(expected) is list:
                assert type(result) is list
                expected = expected[0]
                result = result[0]
            assert result.shape == expected

    @pytest.mark.parametrize("keepdims,expected", [
        (False, (1, 8, 8)),
        (True, (1, 8, 8, 3)),
    ])
    def test__call__if_keepdims_is_(self, keepdims, expected, conv_model):
        saliency = Saliency(conv_model)
        result = saliency(MockScore(), dummy_sample((1, 8, 8, 3)), keepdims=keepdims)
        assert result.shape == expected

    @pytest.mark.parametrize("smooth_samples", [1, 3])
    def test__call__if_smoothing_is_active(self, smooth_samples, conv_model):
        saliency = Saliency(conv_model)
        result = saliency(MockScore(), dummy_sample((1, 8, 8, 3)), smooth_samples=smooth_samples)
        assert result.shape == (1, 8, 8)

    def test__call__if_model_has_only_dense_layers(self, dense_model):
        saliency = Saliency(dense_model)
        result = saliency(MockScore(), dummy_sample((3, )), keepdims=True)
        assert result.shape == (1, 3)

    def test__call__with_categorical_score(self, conv_model):
        # Release v.0.6.0@dev(May 22 2021):
        #   Add this case to test Saliency with CategoricalScore.
        def model_modifier(model):
            model.layers[-1].activation = tf.keras.activations.linear

        score = CategoricalScore([1, 1, 0, 1])
        X = dummy_sample((4, 8, 8, 3))
        saliency = Saliency(conv_model, model_modifier=model_modifier, clone=False)
        result = saliency(score, X)
        assert result.shape == (4, 8, 8)


class TestSaliencyWithMultipleInputsModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        (MockTupleOfScore(), does_not_raise()),
        (MockListOfScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_inputs_model):
        saliency = Saliency(multiple_inputs_model)
        with expectation:
            result = saliency(scores, [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
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
        saliency = Saliency(multiple_inputs_model)
        with expectation:
            result = saliency(MockScore(), seed_input)
            assert len(result) == 2
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)

    @pytest.mark.parametrize("keepdims,expected", [
        (False, [(1, 8, 8), (1, 10, 10)]),
        (True, [(1, 8, 8, 3), (1, 10, 10, 3)]),
    ])
    def test__call__if_keepdims_is_(self, keepdims, expected, multiple_inputs_model):
        saliency = Saliency(multiple_inputs_model)
        result = saliency(MockScore(), [dummy_sample(
            (1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))],
                          keepdims=keepdims)
        assert len(result) == 2
        assert result[0].shape == expected[0]
        assert result[1].shape == expected[1]


class TestSaliencyWithMultipleOutputsModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        ([None], pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
        ([MockScore(), None], pytest.raises(ValueError)),
        ([MockScore(), MockScore()], does_not_raise()),
        ([MockTupleOfScore(), MockTupleOfScore()], does_not_raise()),
        ([MockListOfScore(), MockListOfScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_outputs_model):
        saliency = Saliency(multiple_outputs_model)
        with expectation:
            result = saliency(scores, dummy_sample((1, 8, 8, 3)))
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
        saliency = Saliency(multiple_outputs_model)
        with expectation:
            result = saliency(MockScore(), seed_input)
            if type(expected) is list:
                assert type(result) is list
                expected = expected[0]
                result = result[0]
            assert result.shape == expected


class TestSaliencyWithMultipleIOModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        ([None], pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
        ([MockScore(), None], pytest.raises(ValueError)),
        ([MockScore(), MockScore()], does_not_raise()),
        ([MockTupleOfScore(), MockTupleOfScore()], does_not_raise()),
        ([MockListOfScore(), MockListOfScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_io_model):
        saliency = Saliency(multiple_io_model)
        with expectation:
            result = saliency(scores, [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
            assert len(result) == 2
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)

    @pytest.mark.parametrize("seed_input,expectation", [
        (None, pytest.raises(ValueError)),
        (dummy_sample((1, 8, 8, 3)), pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3))], pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], does_not_raise()),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expectation, multiple_io_model):
        saliency = Saliency(multiple_io_model)
        with expectation:
            result = saliency(MockScore(), seed_input)
            assert len(result) == 2
            assert result[0].shape == (1, 8, 8)
            assert result[1].shape == (1, 10, 10)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
class TestSaliencyWithMixedPrecision():
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
        model = mock_conv_model_with_flot32_output()
        self._test_for_single_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("float32_output.h5")
        model.save(path)
        set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model)

    def _test_for_single_io(self, model):
        saliency = Saliency(model)
        result = saliency(MockScore(), dummy_sample((1, 8, 8, 3)))
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
        saliency = Saliency(model)
        result = saliency(MockScore(), [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))])
        assert len(result) == 2
        assert result[0].shape == (1, 8, 8)
        assert result[1].shape == (1, 10, 10)
