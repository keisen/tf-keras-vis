import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.models import load_model

from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.input_modifiers import Jitter, Rotate2D
from tf_keras_vis.utils.regularizers import Norm, TotalVariation2D
from tf_keras_vis.utils.test import (MockCallback, MockListOfScore, MockScore, MockTupleOfScore,
                                     does_not_raise, dummy_sample, mock_conv_model,
                                     mock_conv_model_with_flot32_output, mock_multiple_io_model)

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import set_global_policy


class TestActivationMaximization():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        (MockTupleOfScore(), does_not_raise()),
        (MockListOfScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        with expectation:
            result = activation_maximization(scores, steps=3)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("seed_input,expected", [
        ([dummy_sample((8, 8, 3))], [(1, 8, 8, 3)]),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8, 3)),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8, 3)]),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(MockScore(), seed_input=seed_input, steps=3)
        if type(expected) is list:
            assert type(result) == list
            result = result[0]
            expected = expected[0]
        assert result.shape == expected

    def test__call__with_callback(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        mock = MockCallback()
        result = activation_maximization(MockScore(), steps=3, callbacks=mock)
        assert result.shape == (1, 8, 8, 3)
        assert mock.on_begin_was_called
        assert mock.on_call_was_called
        assert mock.on_end_was_called

    def test__call__with_gradient_modifier(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(MockScore(), steps=3, gradient_modifier=lambda x: x * 0.0)
        assert result.shape == (1, 8, 8, 3)

    def test__call__if_normalize_gradient_is_True(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(MockScore(), steps=3, normalize_gradient=True)
        assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("input_modifiers,regularizers,expectation", [
        ([Jitter(), Rotate2D()], [TotalVariation2D(), Norm()], does_not_raise()),
        ([Jitter()], [], does_not_raise()),
        ([Rotate2D()], [], does_not_raise()),
        ([], [TotalVariation2D()], does_not_raise()),
        ([], [Norm()], does_not_raise()),
        ([], [], does_not_raise()),
        (None, [], does_not_raise()),
        ([], None, does_not_raise()),
        (None, None, does_not_raise()),
    ])
    def test__call__if_input_modifiers_or_regurarizers_are(self, input_modifiers, regularizers,
                                                           expectation, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        with expectation:
            result = activation_maximization(MockScore(),
                                             input_modifiers=input_modifiers,
                                             regularizers=regularizers,
                                             steps=3)
        assert result.shape == (1, 8, 8, 3)


class TestActivationMaximizationWithMultipleInputsModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        (MockTupleOfScore(), does_not_raise()),
        (MockListOfScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_inputs_model):
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with expectation:
            result = activation_maximization(scores, steps=3)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("seed_inputs,expectation", [
        (None, does_not_raise()),
        (dummy_sample((1, 8, 8, 3)), pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3))], pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3)), None], pytest.raises(ValueError)),
        ([None, dummy_sample((1, 10, 10, 3))], pytest.raises(ValueError)),
        ([dummy_sample((8, 8, 3)), dummy_sample((10, 10, 3))], does_not_raise()),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((10, 10, 3))], does_not_raise()),
        ([dummy_sample((8, 8, 3)), dummy_sample((1, 10, 10, 3))], does_not_raise()),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], does_not_raise()),
    ])
    def test__call__if_seed_input_is_(self, seed_inputs, expectation, multiple_inputs_model):
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with expectation:
            result = activation_maximization(MockScore(), steps=3, seed_input=seed_inputs)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)


class TestActivationMaximizationWithMultipleOutputsModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
        ([MockScore(), None], pytest.raises(ValueError)),
        ([MockScore(), MockScore()], does_not_raise()),
        ([MockTupleOfScore(), MockTupleOfScore()], does_not_raise()),
        ([MockListOfScore(), MockListOfScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_outputs_model):
        activation_maximization = ActivationMaximization(multiple_outputs_model)
        with expectation:
            result = activation_maximization(scores, steps=3)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("seed_input,expected", [
        ([dummy_sample((8, 8, 3))], [(1, 8, 8, 3)]),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8, 3)),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8, 3)]),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected, multiple_outputs_model):
        activation_maximization = ActivationMaximization(multiple_outputs_model)
        result = activation_maximization(MockScore(), seed_input=seed_input, steps=3)
        if type(expected) is list:
            assert type(result) == list
            result = result[0]
            expected = expected[0]
        assert result.shape == expected


class TestActivationMaximizationWithMultipleIOModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
        ([MockScore(), None], pytest.raises(ValueError)),
        ([MockScore(), MockScore()], does_not_raise()),
        ([MockTupleOfScore(), MockTupleOfScore()], does_not_raise()),
        ([MockListOfScore(), MockListOfScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, multiple_io_model):
        activation_maximization = ActivationMaximization(multiple_io_model)
        with expectation:
            result = activation_maximization(scores, steps=3)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("seed_inputs,expectation", [
        (None, does_not_raise()),
        (dummy_sample((1, 8, 8, 3)), pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3))], pytest.raises(ValueError)),
        ([dummy_sample((1, 8, 8, 3)), None], pytest.raises(ValueError)),
        ([None, dummy_sample((1, 10, 10, 3))], pytest.raises(ValueError)),
        ([dummy_sample((8, 8, 3)), dummy_sample((10, 10, 3))], does_not_raise()),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((10, 10, 3))], does_not_raise()),
        ([dummy_sample((8, 8, 3)), dummy_sample((1, 10, 10, 3))], does_not_raise()),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], does_not_raise()),
    ])
    def test__call__if_seed_input_is_(self, seed_inputs, expectation, multiple_io_model):
        activation_maximization = ActivationMaximization(multiple_io_model)
        with expectation:
            result = activation_maximization(MockScore(), steps=3, seed_input=seed_inputs)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    def test__call__with_inputs_modifiers(self, multiple_io_model):
        activation_maximization = ActivationMaximization(multiple_io_model)
        result = activation_maximization(
            MockScore(),
            steps=3,
            input_modifiers={'input-1': [Jitter(jitter=8), Rotate2D(degree=3)]})
        assert result[0].shape == (1, 8, 8, 3)
        assert result[1].shape == (1, 10, 10, 3)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
class TestActivationMaximizationWithMixedPrecision():
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
        activation_maximization = ActivationMaximization(model)
        result = activation_maximization(MockScore(), steps=3)
        assert result.shape == (1, 8, 8, 3)

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
        activation_maximization = ActivationMaximization(model)
        result = activation_maximization(MockScore(), steps=3)
        assert result[0].shape == (1, 8, 8, 3)
        assert result[1].shape == (1, 10, 10, 3)

    def test__call__when_reuse_optimizer(self):
        set_global_policy('mixed_float16')
        optimizer = tf.keras.optimizers.RMSprop()
        model = mock_conv_model()
        activation_maximization = ActivationMaximization(model)
        result = activation_maximization(MockScore(), steps=3, optimizer=optimizer)
        assert result.shape == (1, 8, 8, 3)
        with pytest.raises(ValueError):
            result = activation_maximization(MockScore(), steps=3, optimizer=optimizer)
            assert result.shape == (1, 8, 8, 3)


class TestActivationMaximizationWithDenseModel():
    @pytest.mark.parametrize("scores,expectation", [
        (None, pytest.raises(ValueError)),
        (MockScore(), does_not_raise()),
        (MockTupleOfScore(), does_not_raise()),
        (MockListOfScore(), does_not_raise()),
        ([MockScore()], does_not_raise()),
    ])
    def test__call__if_score_is_(self, scores, expectation, dense_model):
        activation_maximization = ActivationMaximization(dense_model)
        with expectation:
            result = activation_maximization(scores,
                                             input_modifiers=[],
                                             regularizers=[Norm(10.)],
                                             steps=3)
            assert result.shape == (1, 8)

    @pytest.mark.parametrize("seed_input,expected", [
        ([dummy_sample((8, ))], [(1, 8)]),
        (dummy_sample((1, 8)), (1, 8)),
        ([dummy_sample((1, 8))], [(1, 8)]),
    ])
    def test__call__if_seed_input_is_(self, seed_input, expected, dense_model):
        activation_maximization = ActivationMaximization(dense_model)
        result = activation_maximization(MockScore(),
                                         seed_input=seed_input,
                                         input_modifiers=[],
                                         regularizers=[Norm(10.)],
                                         steps=3)
        if type(expected) is list:
            assert type(result) == list
            result = result[0]
            expected = expected[0]
        assert result.shape == expected

    @pytest.mark.parametrize("input_modifiers,regularizers,expectation", [
        ([Jitter(), Rotate2D()], [TotalVariation2D(), Norm()], pytest.raises(ValueError)),
        ([Jitter()], [], pytest.raises(ValueError)),
        ([Rotate2D()], [], pytest.raises(ValueError)),
        ([], [TotalVariation2D()], pytest.raises(ValueError)),
        ([], [Norm()], does_not_raise()),
        ([], [], does_not_raise()),
    ])
    def test__call__if_input_modifiers_or_regurarizers_are(self, input_modifiers, regularizers,
                                                           expectation, dense_model):
        activation_maximization = ActivationMaximization(dense_model)
        with expectation:
            result = activation_maximization(MockScore(),
                                             input_modifiers=input_modifiers,
                                             regularizers=regularizers,
                                             steps=3)
            assert result.shape == (1, 8)
