import numpy as np
import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.models import load_model

from tf_keras_vis.activation_maximization import \
    ActivationMaximization as CurrentActivationMaximization  # noqa: E501
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate, Scale
from tf_keras_vis.activation_maximization.legacy import \
    ActivationMaximization as LegacyActivationMaximization  # noqa: E501
from tf_keras_vis.activation_maximization.regularizers import Norm, TotalVariation2D
from tf_keras_vis.utils import listify
from tf_keras_vis.utils.regularizers import LegacyRegularizer
from tf_keras_vis.utils.regularizers import Norm as LegacyNorm
from tf_keras_vis.utils.regularizers import TotalVariation2D as LegacyTotalVariation2D
from tf_keras_vis.utils.scores import BinaryScore, CategoricalScore
from tf_keras_vis.utils.test import (NO_ERROR, MockCallback, MockLegacyCallback, assert_raises,
                                     dummy_sample, mock_conv_model,
                                     mock_conv_model_with_float32_output, mock_multiple_io_model,
                                     score_with_list, score_with_tuple)

ActivationMaximization = CurrentActivationMaximization


@pytest.fixture(scope='function',
                params=[CurrentActivationMaximization, LegacyActivationMaximization])
def legacy(request):
    global ActivationMaximization
    ActivationMaximization = request.param
    yield
    ActivationMaximization = CurrentActivationMaximization


class TestActivationMaximization():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        (score_with_tuple, NO_ERROR),
        (score_with_list, NO_ERROR),
        (CategoricalScore(0), NO_ERROR),
        ([None], ValueError),
        ([score_with_tuple], NO_ERROR),
        ([score_with_list], NO_ERROR),
        ([CategoricalScore(0)], NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_score_is_(self, scores, expected_error, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        with assert_raises(expected_error):
            result = activation_maximization(scores)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("seed_input,expected", [
        (None, (1, 8, 8, 3)),
        (dummy_sample((8, 8, 3)), (1, 8, 8, 3)),
        ([dummy_sample((8, 8, 3))], [(1, 8, 8, 3)]),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8, 3)),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8, 3)]),
        (dummy_sample((4, 8, 8, 3)), (4, 8, 8, 3)),
        ([dummy_sample((4, 8, 8, 3))], [(4, 8, 8, 3)]),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_seed_input_is_(self, seed_input, expected, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(CategoricalScore(0), seed_input=seed_input)
        if type(expected) is list:
            assert type(result) == list
            result = result[0]
            expected = expected[0]
        assert result.shape == expected

    @pytest.mark.parametrize("input_range,,expected_error", [
        (None, NO_ERROR),
        ((None, None), NO_ERROR),
        ((0, None), NO_ERROR),
        ((None, 255), NO_ERROR),
        ((0, 255), NO_ERROR),
        ((-1.0, 1.0), NO_ERROR),
        ((-1.0, 255), TypeError),
        ((0, 1.0), TypeError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test_call__if_input_range_is_(self, input_range, expected_error, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), input_range=input_range)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("input_modifiers,expected_error", [
        (None, NO_ERROR),
        (Jitter(), NO_ERROR),
        ([], NO_ERROR),
        ([None], TypeError),
        ([Jitter()], NO_ERROR),
        ([Jitter(), None], TypeError),
        ([None, Jitter()], TypeError),
        ([Jitter(), Rotate(), Scale()], NO_ERROR),
        ([[]], NO_ERROR),
        ([[None]], TypeError),
        ([[Jitter()]], NO_ERROR),
        ([[Jitter(), None]], TypeError),
        ([[None, Jitter()]], TypeError),
        ([[Jitter(), Rotate(), Scale()]], NO_ERROR),
        ([[Jitter(), Rotate(), Scale()], [Jitter(), Rotate(), Scale()]], ValueError),
        (dict(input_1=None), NO_ERROR),
        (dict(input_1=Jitter()), NO_ERROR),
        (dict(input_1=[]), NO_ERROR),
        (dict(input_1=[None]), TypeError),
        (dict(input_1=[Jitter()]), NO_ERROR),
        (dict(input_1=[Jitter(), None]), TypeError),
        (dict(input_1=[None, Jitter()]), TypeError),
        (dict(input_1=[Jitter(), Rotate(), Scale()]), NO_ERROR),
        (dict(input_2=[Jitter(), Rotate(), Scale()]), ValueError),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[Jitter(), Rotate(),
                                                              Scale()]), ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_input_modifiers_are_(self, input_modifiers, expected_error, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), input_modifiers=input_modifiers)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("regularizers,expected_error", [
        (None, NO_ERROR),
        (TotalVariation2D(), NO_ERROR),
        (LegacyTotalVariation2D(), NO_ERROR),
        ([], NO_ERROR),
        ([None], TypeError),
        ([TotalVariation2D()], NO_ERROR),
        ([LegacyNorm()], NO_ERROR),
        ([TotalVariation2D(), None], TypeError),
        ([None, TotalVariation2D()], TypeError),
        ([TotalVariation2D(), LegacyTotalVariation2D()], ValueError),
        ([TotalVariation2D(), Norm()], NO_ERROR),
        (dict(input_1=None), NO_ERROR),
        (dict(input_1=[]), NO_ERROR),
        (dict(input_1=TotalVariation2D()), NO_ERROR),
        (dict(input_1=LegacyTotalVariation2D()), ValueError),
        (dict(input_2=None), ValueError),
        (dict(input_2=[]), ValueError),
        (dict(input_2=TotalVariation2D()), ValueError),
        (dict(input_2=LegacyTotalVariation2D()), ValueError),
        (dict(input_1=TotalVariation2D(), input_2=TotalVariation2D()), ValueError),
        (dict(input_1=LegacyTotalVariation2D(), input_2=TotalVariation2D()), ValueError),
        (dict(input_1=LegacyTotalVariation2D(), input_2=LegacyTotalVariation2D()), ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_regularizer_is_(self, regularizers, expected_error, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), regularizers=regularizers)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("regularizer_container", [list, tuple, dict])
    @pytest.mark.parametrize("regularizers,expected_error", [
        ([[]], NO_ERROR),
        ([[None]], TypeError),
        ([[TotalVariation2D()]], NO_ERROR),
        ([[LegacyTotalVariation2D()]], ValueError),
        ([[TotalVariation2D(), None]], TypeError),
        ([[None, TotalVariation2D()]], TypeError),
        ([[TotalVariation2D(), Norm()]], NO_ERROR),
        ([[TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[LegacyNorm(), LegacyTotalVariation2D()]], ValueError),
        ([[], [Norm()]], ValueError),
        ([[None], [Norm()]], ValueError),
        ([[TotalVariation2D()], [Norm()]], ValueError),
        ([[LegacyTotalVariation2D()], [Norm()]], ValueError),
        ([[TotalVariation2D(), None], [Norm()]], ValueError),
        ([[None, TotalVariation2D()], [Norm()]], ValueError),
        ([[TotalVariation2D(), Norm()], [Norm()]], ValueError),
        ([[TotalVariation2D(), LegacyTotalVariation2D()], [Norm()]], ValueError),
        ([[Norm()], []], ValueError),
        ([[Norm()], [None]], ValueError),
        ([[Norm()], [TotalVariation2D()]], ValueError),
        ([[Norm()], [LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [TotalVariation2D(), None]], ValueError),
        ([[Norm()], [None, TotalVariation2D()]], ValueError),
        ([[Norm()], [TotalVariation2D(), Norm()]], ValueError),
        ([[Norm()], [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[TotalVariation2D(), LegacyTotalVariation2D()], None], ValueError),
        ([None, [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_regularizers_are_(self, regularizer_container, regularizers, expected_error,
                                         conv_model):
        if regularizer_container is tuple:
            regularizers = tuple(regularizers)
        if regularizer_container is dict:
            regularizers = zip(['input_1', 'input_2'], regularizers)
            regularizers = dict(regularizers)
            has_legacy = ((isinstance(r, LegacyRegularizer) for r in listify(_regularizers))
                          for _regularizers in regularizers.values())
            if any((any(f) for f in has_legacy)):
                expected_error = ValueError
        activation_maximization = ActivationMaximization(conv_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), regularizers=regularizers)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_normalize_gradient_is_True(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(CategoricalScore(0), normalize_gradient=True)
        assert result.shape == (1, 8, 8, 3)

    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__with_gradient_modifier(self, conv_model):
        activation_maximization = ActivationMaximization(conv_model)
        result = activation_maximization(CategoricalScore(0), gradient_modifier=lambda x: x * 0.0)
        assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("is_legacy", [False, True])
    @pytest.mark.parametrize("callbacks,expected,expected_error", [
        (None, [], NO_ERROR),
        (MockCallback(), [True], NO_ERROR),
        ([], [], NO_ERROR),
        ([MockCallback()], [True], NO_ERROR),
        ([MockCallback(), MockCallback()], [True, True], NO_ERROR),
        ([MockCallback(raise_error_on_begin=True),
          MockCallback()], [False, False], ValueError),
        ([MockCallback(), MockCallback(raise_error_on_begin=True)], [True, False], ValueError),
        ([MockCallback(raise_error_on_call=True),
          MockCallback()], [True, True], ValueError),
        ([MockCallback(), MockCallback(raise_error_on_call=True)], [True, True], ValueError),
        ([MockCallback(raise_error_on_end=True),
          MockCallback()], [True, True], ValueError),
        ([MockCallback(raise_error_on_end=True),
          MockCallback()], [True, True], ValueError),
        ([MockCallback(), MockCallback(raise_error_on_end=True)], [True, True], ValueError),
        ([MockCallback(raise_error_on_end=True),
          MockCallback(raise_error_on_end=True)], [True, True], ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__with_callbacks(self, is_legacy, callbacks, expected, expected_error,
                                   conv_model):
        if is_legacy:
            if isinstance(callbacks, MockCallback):
                callbacks = MockLegacyCallback(callbacks)
            if isinstance(callbacks, list):
                callbacks = [
                    MockLegacyCallback(c) if isinstance(c, MockCallback) else c for c in callbacks
                ]
        activation_maximization = ActivationMaximization(conv_model)
        try:
            result = activation_maximization(CategoricalScore(0), callbacks=callbacks)
            assert expected_error == NO_ERROR
            assert result.shape == (1, 8, 8, 3)
        except ValueError:
            assert expected_error != NO_ERROR
        finally:
            for c, e in zip(listify(callbacks), expected):
                if is_legacy:
                    assert c.callback.on_end_was_called == e
                else:
                    assert c.on_end_was_called == e

    @pytest.mark.parametrize("activation_modifiers,modified,expected_error", [
        (None, False, NO_ERROR),
        (lambda x: np.ones(x.shape, np.float), True, NO_ERROR),
        (dict(input_1=None), False, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float)), True, NO_ERROR),
        (dict(input_2=lambda x: np.ones(x.shape, np.float)), False, ValueError),
        (dict(input_1=lambda x: np.ones(x.shape, np.float),
              input_2=lambda x: np.ones(x.shape, np.float)), False, ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__with_activation_modifiers(self, activation_modifiers, modified, expected_error,
                                              conv_model):
        seed_inputs = dummy_sample((1, 8, 8, 3))
        activation_maximization = ActivationMaximization(conv_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), seed_input=seed_inputs)
            assert not np.all(result == 0.0)
            result = activation_maximization(CategoricalScore(0),
                                             seed_input=seed_inputs,
                                             activation_modifiers=activation_modifiers)
            if modified:
                assert np.all(result == 1.0)
            else:
                assert not np.all(result == 1.0)


class TestActivationMaximizationWithMultipleInputsModel():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        (score_with_tuple, NO_ERROR),
        (score_with_list, NO_ERROR),
        (CategoricalScore(0), NO_ERROR),
        ([None], ValueError),
        ([score_with_tuple], NO_ERROR),
        ([score_with_list], NO_ERROR),
        ([CategoricalScore(0)], NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_score_is_(self, scores, expected_error, multiple_inputs_model):
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with assert_raises(expected_error):
            result = activation_maximization(scores)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("seed_inputs,expected_error", [
        (None, NO_ERROR),
        (dummy_sample((1, 8, 8, 3)), ValueError),
        ([dummy_sample((1, 8, 8, 3))], ValueError),
        ([dummy_sample((1, 8, 8, 3)), None], ValueError),
        ([None, dummy_sample((1, 10, 10, 3))], ValueError),
        ([dummy_sample((8, 8, 3)), dummy_sample((10, 10, 3))], NO_ERROR),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((10, 10, 3))], NO_ERROR),
        ([dummy_sample((8, 8, 3)), dummy_sample((1, 10, 10, 3))], NO_ERROR),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], NO_ERROR),
        ([dummy_sample((4, 8, 8, 3)), dummy_sample((4, 10, 10, 3))], NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_seed_input_is_(self, seed_inputs, expected_error, multiple_inputs_model):
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), seed_input=seed_inputs)
            if seed_inputs is not None and seed_inputs[0].shape[0] == 4:
                assert result[0].shape == (4, 8, 8, 3)
                assert result[1].shape == (4, 10, 10, 3)
            else:
                assert result[0].shape == (1, 8, 8, 3)
                assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("input_modifiers,expected_error", [
        (None, NO_ERROR),
        (Jitter(), NO_ERROR),
        ([], NO_ERROR),
        ([None], TypeError),
        ([Jitter()], NO_ERROR),
        ([Jitter(), None], TypeError),
        ([None, Jitter()], TypeError),
        ([Jitter(), Rotate(), Scale()], NO_ERROR),
        ([[]], NO_ERROR),
        ([[None]], TypeError),
        ([[Jitter()]], NO_ERROR),
        ([[Jitter(), None]], TypeError),
        ([[None, Jitter()]], TypeError),
        ([[Jitter(), Rotate(), Scale()]], NO_ERROR),
        ([[Jitter(), Rotate(), Scale()], []], NO_ERROR),
        ([[Jitter(), Rotate(), Scale()], [None]], TypeError),
        ([[Jitter(), Rotate(), Scale()], [Jitter()]], NO_ERROR),
        ([[Jitter(), Rotate(), Scale()], [Jitter(), None]], TypeError),
        ([[Jitter(), Rotate(), Scale()], [None, Jitter()]], TypeError),
        ([[Jitter(), Rotate(), Scale()], [Jitter(), Rotate(), Scale()]], NO_ERROR),
        (dict(input_1=None), NO_ERROR),
        (dict(input_1=Jitter()), NO_ERROR),
        (dict(input_1=[]), NO_ERROR),
        (dict(input_1=[None]), TypeError),
        (dict(input_1=[Jitter()]), NO_ERROR),
        (dict(input_1=[Jitter(), None]), TypeError),
        (dict(input_1=[None, Jitter()]), TypeError),
        (dict(input_1=[Jitter(), Rotate(), Scale()]), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=None), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=Jitter()), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[]), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[None]), TypeError),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[Jitter()]), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[Jitter(), None]), TypeError),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[None, Jitter()]), TypeError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_input_modifiers_are_(self, input_modifiers, expected_error,
                                            multiple_inputs_model):
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), input_modifiers=input_modifiers)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("regularizers,expected_error", [
        (None, NO_ERROR),
        (TotalVariation2D(), NO_ERROR),
        (LegacyTotalVariation2D(), NO_ERROR),
        ([], NO_ERROR),
        ([None], TypeError),
        ([TotalVariation2D()], NO_ERROR),
        ([LegacyTotalVariation2D()], NO_ERROR),
        ([TotalVariation2D(), None], TypeError),
        ([None, TotalVariation2D()], TypeError),
        ([TotalVariation2D(), LegacyTotalVariation2D()], ValueError),
        ([TotalVariation2D(), Norm()], NO_ERROR),
        ([LegacyTotalVariation2D(), LegacyNorm()], NO_ERROR),
        (dict(input_1=None), NO_ERROR),
        (dict(input_1=[]), NO_ERROR),
        (dict(input_1=TotalVariation2D()), NO_ERROR),
        (dict(input_1=LegacyTotalVariation2D()), ValueError),
        (dict(input_2=None), NO_ERROR),
        (dict(input_2=[]), NO_ERROR),
        (dict(input_2=TotalVariation2D()), NO_ERROR),
        (dict(input_2=LegacyTotalVariation2D()), ValueError),
        (dict(input_3=None), ValueError),
        (dict(input_3=[]), ValueError),
        (dict(input_3=TotalVariation2D()), ValueError),
        (dict(input_3=LegacyTotalVariation2D()), ValueError),
        (dict(input_1=TotalVariation2D(), input_2=TotalVariation2D()), NO_ERROR),
        (dict(input_1=LegacyTotalVariation2D(), input_2=TotalVariation2D()), ValueError),
        (dict(input_1=LegacyTotalVariation2D(), input_2=LegacyTotalVariation2D()), ValueError),
        (dict(input_1=TotalVariation2D(), input_2=TotalVariation2D(),
              input_3=TotalVariation2D()), ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_regularizer_is_(self, regularizers, expected_error, multiple_inputs_model):
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), regularizers=regularizers)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("regularizer_container", [list, tuple, dict])
    @pytest.mark.parametrize("regularizers,expected_error", [
        ([[]], NO_ERROR),
        ([[None]], TypeError),
        ([[TotalVariation2D()]], NO_ERROR),
        ([[LegacyTotalVariation2D()]], ValueError),
        ([[TotalVariation2D(), None]], TypeError),
        ([[None, TotalVariation2D()]], TypeError),
        ([[TotalVariation2D(), Norm()]], NO_ERROR),
        ([[TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[LegacyNorm(), LegacyTotalVariation2D()]], ValueError),
        ([[], [Norm()]], NO_ERROR),
        ([[None], [Norm()]], TypeError),
        ([[TotalVariation2D()], [Norm()]], NO_ERROR),
        ([[LegacyTotalVariation2D()], [Norm()]], ValueError),
        ([[TotalVariation2D(), None], [Norm()]], TypeError),
        ([[None, TotalVariation2D()], [Norm()]], TypeError),
        ([[TotalVariation2D(), Norm()], [Norm()]], NO_ERROR),
        ([[TotalVariation2D(), LegacyTotalVariation2D()], [Norm()]], ValueError),
        ([[Norm()], []], NO_ERROR),
        ([[Norm()], [None]], TypeError),
        ([[Norm()], [TotalVariation2D()]], NO_ERROR),
        ([[Norm()], [LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [TotalVariation2D(), None]], TypeError),
        ([[Norm()], [None, TotalVariation2D()]], TypeError),
        ([[Norm()], [TotalVariation2D(), Norm()]], NO_ERROR),
        ([[Norm()], [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[TotalVariation2D(), LegacyTotalVariation2D()], None], ValueError),
        ([None, [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], []], ValueError),
        ([[Norm()], [Norm()], [None]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], [LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D(), None]], ValueError),
        ([[Norm()], [Norm()], [None, TotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D(), Norm()]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [TotalVariation2D(), LegacyTotalVariation2D()], None], ValueError),
        ([None, [Norm()], [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_regularizers_are_(self, regularizer_container, regularizers, expected_error,
                                         multiple_inputs_model):
        if regularizer_container is tuple:
            regularizers = tuple(regularizers)
        if regularizer_container is dict:
            regularizers = zip(['input_1', 'input_2', 'input_3'], regularizers)
            regularizers = dict(regularizers)
            has_legacy = ((isinstance(r, LegacyRegularizer) for r in listify(_regularizers))
                          for _regularizers in regularizers.values())
            if any((any(f) for f in has_legacy)):
                expected_error = ValueError
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), regularizers=regularizers)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("activation_modifiers,modified_0,modified_1,expected_error", [
        (None, False, False, NO_ERROR),
        (lambda x: np.ones(x.shape, np.float), True, False, NO_ERROR),
        (dict(input_1=None), False, False, NO_ERROR),
        (dict(input_2=None), False, False, NO_ERROR),
        (dict(input_1=None, input_2=None), False, False, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float)), True, False, NO_ERROR),
        (dict(input_2=lambda x: np.ones(x.shape, np.float)), False, True, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float), input_2=None), True, False, NO_ERROR),
        (dict(input_1=None, input_2=lambda x: np.ones(x.shape, np.float)), False, True, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float),
              input_2=lambda x: np.ones(x.shape, np.float)), True, True, NO_ERROR),
        (dict(input_1=None, input_2=None,
              input_3=lambda x: np.ones(x.shape, np.float)), False, False, ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__with_activation_modifiers(self, activation_modifiers, modified_0, modified_1,
                                              expected_error, multiple_inputs_model):
        seed_inputs = [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))]
        activation_maximization = ActivationMaximization(multiple_inputs_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0), seed_input=seed_inputs)
            assert not np.all(result[0] == 1.0)
            assert not np.all(result[1] == 1.0)
            result = activation_maximization(CategoricalScore(0),
                                             seed_input=seed_inputs,
                                             activation_modifiers=activation_modifiers)
            if modified_0:
                assert np.all(result[0] == 1.0)
            else:
                assert not np.all(result[0] == 1.0)
            if modified_1:
                assert np.all(result[1] == 1.0)
            else:
                assert not np.all(result[1] == 1.0)


class TestActivationMaximizationWithMultipleOutputsModel():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        (score_with_tuple, ValueError),
        (score_with_list, ValueError),
        (CategoricalScore(0), ValueError),
        ([None], ValueError),
        ([score_with_tuple], ValueError),
        ([score_with_list], ValueError),
        ([CategoricalScore(0)], ValueError),
        ([CategoricalScore(0), None], ValueError),
        ([CategoricalScore(0), score_with_tuple], NO_ERROR),
        ([CategoricalScore(0), score_with_list], NO_ERROR),
        ([CategoricalScore(0), BinaryScore(False)], NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_score_is_(self, scores, expected_error, multiple_outputs_model):
        activation_maximization = ActivationMaximization(multiple_outputs_model)
        with assert_raises(expected_error):
            result = activation_maximization(scores)
            assert result.shape == (1, 8, 8, 3)

    @pytest.mark.parametrize("seed_input,expected", [
        (None, (1, 8, 8, 3)),
        (dummy_sample((8, 8, 3)), (1, 8, 8, 3)),
        ([dummy_sample((8, 8, 3))], [(1, 8, 8, 3)]),
        (dummy_sample((1, 8, 8, 3)), (1, 8, 8, 3)),
        ([dummy_sample((1, 8, 8, 3))], [(1, 8, 8, 3)]),
        (dummy_sample((4, 8, 8, 3)), (4, 8, 8, 3)),
        ([dummy_sample((4, 8, 8, 3))], [(4, 8, 8, 3)]),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_seed_input_is_(self, seed_input, expected, multiple_outputs_model):
        activation_maximization = ActivationMaximization(multiple_outputs_model)
        result = activation_maximization(
            [CategoricalScore(1), BinaryScore(False)], seed_input=seed_input)
        if type(expected) is list:
            assert type(result) == list
            result = result[0]
            expected = expected[0]
        assert result.shape == expected


class TestActivationMaximizationWithMultipleIOModel():
    @pytest.mark.parametrize("scores,expected_error", [
        (None, ValueError),
        (score_with_tuple, ValueError),
        (score_with_list, ValueError),
        (CategoricalScore(0), ValueError),
        ([None], ValueError),
        ([score_with_tuple], ValueError),
        ([score_with_list], ValueError),
        ([CategoricalScore(0)], ValueError),
        ([CategoricalScore(0), None], ValueError),
        ([CategoricalScore(0), score_with_tuple], NO_ERROR),
        ([CategoricalScore(0), score_with_list], NO_ERROR),
        ([CategoricalScore(0), BinaryScore(False)], NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_score_is_(self, scores, expected_error, multiple_io_model):
        activation_maximization = ActivationMaximization(multiple_io_model)
        with assert_raises(expected_error):
            result = activation_maximization(scores)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("seed_inputs,expected_error", [
        (None, NO_ERROR),
        (dummy_sample((1, 8, 8, 3)), ValueError),
        ([dummy_sample((1, 8, 8, 3))], ValueError),
        ([dummy_sample((1, 8, 8, 3)), None], ValueError),
        ([None, dummy_sample((1, 10, 10, 3))], ValueError),
        ([dummy_sample((8, 8, 3)), dummy_sample((10, 10, 3))], NO_ERROR),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((10, 10, 3))], NO_ERROR),
        ([dummy_sample((8, 8, 3)), dummy_sample((1, 10, 10, 3))], NO_ERROR),
        ([dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))], NO_ERROR),
        ([dummy_sample((4, 8, 8, 3)), dummy_sample((4, 10, 10, 3))], NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_seed_input_is_(self, seed_inputs, expected_error, multiple_io_model):
        activation_maximization = ActivationMaximization(multiple_io_model)
        with assert_raises(expected_error):
            result = activation_maximization(
                [CategoricalScore(1), BinaryScore(True)], seed_input=seed_inputs)
            if seed_inputs is not None and seed_inputs[0].shape[0] == 4:
                assert result[0].shape == (4, 8, 8, 3)
                assert result[1].shape == (4, 10, 10, 3)
            else:
                assert result[0].shape == (1, 8, 8, 3)
                assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("input_modifiers,expected_error", [
        (None, NO_ERROR),
        (Jitter(), NO_ERROR),
        ([], NO_ERROR),
        ([None], TypeError),
        ([Jitter()], NO_ERROR),
        ([Jitter(), None], TypeError),
        ([None, Jitter()], TypeError),
        ([Jitter(), Rotate(), Scale()], NO_ERROR),
        ([[]], NO_ERROR),
        ([[None]], TypeError),
        ([[Jitter()]], NO_ERROR),
        ([[Jitter(), None]], TypeError),
        ([[None, Jitter()]], TypeError),
        ([[Jitter(), Rotate(), Scale()]], NO_ERROR),
        ([[Jitter(), Rotate(), Scale()], []], NO_ERROR),
        ([[Jitter(), Rotate(), Scale()], [None]], TypeError),
        ([[Jitter(), Rotate(), Scale()], [Jitter()]], NO_ERROR),
        ([[Jitter(), Rotate(), Scale()], [Jitter(), None]], TypeError),
        ([[Jitter(), Rotate(), Scale()], [None, Jitter()]], TypeError),
        ([[Jitter(), Rotate(), Scale()], [Jitter(), Rotate(), Scale()]], NO_ERROR),
        (dict(input_1=None), NO_ERROR),
        (dict(input_1=Jitter()), NO_ERROR),
        (dict(input_1=[]), NO_ERROR),
        (dict(input_1=[None]), TypeError),
        (dict(input_1=[Jitter()]), NO_ERROR),
        (dict(input_1=[Jitter(), None]), TypeError),
        (dict(input_1=[None, Jitter()]), TypeError),
        (dict(input_1=[Jitter(), Rotate(), Scale()]), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=None), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=Jitter()), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[]), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[None]), TypeError),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[Jitter()]), NO_ERROR),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[Jitter(), None]), TypeError),
        (dict(input_1=[Jitter(), Rotate(), Scale()], input_2=[None, Jitter()]), TypeError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_input_modifiers_are_(self, input_modifiers, expected_error,
                                            multiple_io_model):
        activation_maximization = ActivationMaximization(multiple_io_model)
        with assert_raises(expected_error):
            result = activation_maximization(
                [CategoricalScore(1), BinaryScore(True)], input_modifiers=input_modifiers)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("regularizers,expected_error", [
        (None, NO_ERROR),
        (TotalVariation2D(), NO_ERROR),
        (LegacyTotalVariation2D(), NO_ERROR),
        ([], NO_ERROR),
        ([None], TypeError),
        ([TotalVariation2D()], NO_ERROR),
        ([LegacyTotalVariation2D()], NO_ERROR),
        ([TotalVariation2D(), None], TypeError),
        ([None, TotalVariation2D()], TypeError),
        ([TotalVariation2D(), LegacyTotalVariation2D()], ValueError),
        ([TotalVariation2D(), Norm()], NO_ERROR),
        ([LegacyTotalVariation2D(), LegacyNorm()], NO_ERROR),
        (dict(input_1=None), NO_ERROR),
        (dict(input_1=[]), NO_ERROR),
        (dict(input_1=TotalVariation2D()), NO_ERROR),
        (dict(input_1=LegacyTotalVariation2D()), ValueError),
        (dict(input_2=None), NO_ERROR),
        (dict(input_2=[]), NO_ERROR),
        (dict(input_2=TotalVariation2D()), NO_ERROR),
        (dict(input_2=LegacyTotalVariation2D()), ValueError),
        (dict(input_3=None), ValueError),
        (dict(input_3=[]), ValueError),
        (dict(input_3=TotalVariation2D()), ValueError),
        (dict(input_3=LegacyTotalVariation2D()), ValueError),
        (dict(input_1=TotalVariation2D(), input_2=TotalVariation2D()), NO_ERROR),
        (dict(input_1=LegacyTotalVariation2D(), input_2=TotalVariation2D()), ValueError),
        (dict(input_1=LegacyTotalVariation2D(), input_2=LegacyTotalVariation2D()), ValueError),
        (dict(input_1=TotalVariation2D(), input_2=TotalVariation2D(),
              input_3=TotalVariation2D()), ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_regularizer_is_(self, regularizers, expected_error, multiple_io_model):
        activation_maximization = ActivationMaximization(multiple_io_model)
        with assert_raises(expected_error):
            result = activation_maximization(
                [CategoricalScore(0), BinaryScore(True)], regularizers=regularizers)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("regularizer_container", [list, tuple, dict])
    @pytest.mark.parametrize("regularizers,expected_error", [
        ([[]], NO_ERROR),
        ([[None]], TypeError),
        ([[TotalVariation2D()]], NO_ERROR),
        ([[LegacyTotalVariation2D()]], ValueError),
        ([[TotalVariation2D(), None]], TypeError),
        ([[None, TotalVariation2D()]], TypeError),
        ([[TotalVariation2D(), Norm()]], NO_ERROR),
        ([[TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[LegacyNorm(), LegacyTotalVariation2D()]], ValueError),
        ([[], [Norm()]], NO_ERROR),
        ([[None], [Norm()]], TypeError),
        ([[TotalVariation2D()], [Norm()]], NO_ERROR),
        ([[LegacyTotalVariation2D()], [Norm()]], ValueError),
        ([[TotalVariation2D(), None], [Norm()]], TypeError),
        ([[None, TotalVariation2D()], [Norm()]], TypeError),
        ([[TotalVariation2D(), Norm()], [Norm()]], NO_ERROR),
        ([[TotalVariation2D(), LegacyTotalVariation2D()], [Norm()]], ValueError),
        ([[Norm()], []], NO_ERROR),
        ([[Norm()], [None]], TypeError),
        ([[Norm()], [TotalVariation2D()]], NO_ERROR),
        ([[Norm()], [LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [TotalVariation2D(), None]], TypeError),
        ([[Norm()], [None, TotalVariation2D()]], TypeError),
        ([[Norm()], [TotalVariation2D(), Norm()]], NO_ERROR),
        ([[Norm()], [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[TotalVariation2D(), LegacyTotalVariation2D()], None], ValueError),
        ([None, [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], []], ValueError),
        ([[Norm()], [Norm()], [None]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], [LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D(), None]], ValueError),
        ([[Norm()], [Norm()], [None, TotalVariation2D()]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D(), Norm()]], ValueError),
        ([[Norm()], [Norm()], [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
        ([[Norm()], [TotalVariation2D(), LegacyTotalVariation2D()], None], ValueError),
        ([None, [Norm()], [TotalVariation2D(), LegacyTotalVariation2D()]], ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_regularizers_are_(self, regularizer_container, regularizers, expected_error,
                                         multiple_io_model):
        if regularizer_container is tuple:
            regularizers = tuple(regularizers)
        if regularizer_container is dict:
            regularizers = zip(['input_1', 'input_2', 'input_3'], regularizers)
            regularizers = dict(regularizers)
            has_legacy = ((isinstance(r, LegacyRegularizer) for r in listify(_regularizers))
                          for _regularizers in regularizers.values())
            if any((any(f) for f in has_legacy)):
                expected_error = ValueError
        activation_maximization = ActivationMaximization(multiple_io_model)
        with assert_raises(expected_error):
            result = activation_maximization(
                [CategoricalScore(0), BinaryScore(True)], regularizers=regularizers)
            assert result[0].shape == (1, 8, 8, 3)
            assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.parametrize("activation_modifiers,modified_0,modified_1,expected_error", [
        (None, False, False, NO_ERROR),
        (lambda x: np.ones(x.shape, np.float), True, False, NO_ERROR),
        (dict(input_1=None), False, False, NO_ERROR),
        (dict(input_2=None), False, False, NO_ERROR),
        (dict(input_1=None, input_2=None), False, False, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float)), True, False, NO_ERROR),
        (dict(input_2=lambda x: np.ones(x.shape, np.float)), False, True, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float), input_2=None), True, False, NO_ERROR),
        (dict(input_1=None, input_2=lambda x: np.ones(x.shape, np.float)), False, True, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float),
              input_2=lambda x: np.ones(x.shape, np.float)), True, True, NO_ERROR),
        (dict(input_1=None, input_2=None,
              input_3=lambda x: np.ones(x.shape, np.float)), False, False, ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__with_activation_modifiers(self, activation_modifiers, modified_0, modified_1,
                                              expected_error, multiple_io_model):
        seed_inputs = [dummy_sample((1, 8, 8, 3)), dummy_sample((1, 10, 10, 3))]
        activation_maximization = ActivationMaximization(multiple_io_model)
        with assert_raises(expected_error):
            result = activation_maximization(
                [CategoricalScore(1), BinaryScore(True)], seed_input=seed_inputs)
            assert not np.all(result[0] == 1.0)
            assert not np.all(result[1] == 1.0)
            result = activation_maximization(
                [CategoricalScore(1), BinaryScore(True)],
                seed_input=seed_inputs,
                activation_modifiers=activation_modifiers)
            if modified_0:
                assert np.all(result[0] == 1.0)
            else:
                assert not np.all(result[0] == 1.0)
            if modified_1:
                assert np.all(result[1] == 1.0)
            else:
                assert not np.all(result[1] == 1.0)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled only when tensorflow version is 2.4.0+.")
class TestMixedPrecision():
    @pytest.mark.usefixtures("legacy")
    def test__call__with_single_io(self, tmpdir):
        # Create and save lower precision model
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model = mock_conv_model()
        self._test_for_single_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("single_io.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_single_io(model)
        # Load and test lower precision model on full precision environment
        tf.keras.mixed_precision.set_global_policy('float32')
        model = load_model(path)
        self._test_for_single_io(model)

    @pytest.mark.usefixtures("legacy")
    def test__call__with_float32_output_model(self, tmpdir):
        # Create and save lower precision model
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model = mock_conv_model_with_float32_output()
        self._test_for_single_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("float32_output.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_single_io(model)
        tf.keras.mixed_precision.set_global_policy('float32')
        # Load and test lower precision model on full precision environment
        model = load_model(path)
        self._test_for_single_io(model)

    def _test_for_single_io(self, model):
        activation_maximization = ActivationMaximization(model)
        result = activation_maximization(CategoricalScore(1))
        assert result.shape == (1, 8, 8, 3)

    @pytest.mark.usefixtures("legacy")
    def test__call__with_multiple_io(self, tmpdir):
        # Create and save lower precision model
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model = mock_multiple_io_model()
        self._test_for_multiple_io(model)
        path = tmpdir.mkdir("tf-keras-vis").join("multiple_io.h5")
        model.save(path)
        # Load and test lower precision model on lower precision environment
        model = load_model(path)
        self._test_for_multiple_io(model)
        # Load and test lower precision model on full precision environment
        tf.keras.mixed_precision.set_global_policy('float32')
        model = load_model(path)
        self._test_for_multiple_io(model)

    def _test_for_multiple_io(self, model):
        activation_maximization = ActivationMaximization(model)
        result = activation_maximization([CategoricalScore(1), BinaryScore(False)])
        assert result[0].shape == (1, 8, 8, 3)
        assert result[1].shape == (1, 10, 10, 3)

    @pytest.mark.usefixtures("legacy")
    def test__call__when_reuse_optimizer(self):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        optimizer = tf.keras.optimizers.RMSprop()
        model = mock_conv_model()
        activation_maximization = ActivationMaximization(model)
        with assert_raises(NO_ERROR):
            result = activation_maximization(CategoricalScore(0), optimizer=optimizer)
            assert result.shape == (1, 8, 8, 3)
        with assert_raises(ValueError):
            result = activation_maximization(CategoricalScore(0), optimizer=optimizer)
            assert result.shape == (1, 8, 8, 3)


class TestActivationMaximizationWithDenseModel():
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
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_score_is_(self, scores, expected_error, dense_model):
        activation_maximization = ActivationMaximization(dense_model)
        with assert_raises(expected_error):
            result = activation_maximization(scores, input_modifiers=None, regularizers=None)
            assert result.shape == (1, 8)

    @pytest.mark.parametrize("seed_input,expected", [
        (None, (1, 8)),
        (dummy_sample((8, )), (1, 8)),
        ([dummy_sample((8, ))], [(1, 8)]),
        (dummy_sample((1, 8)), (1, 8)),
        ([dummy_sample((1, 8))], [(1, 8)]),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_seed_input_is_(self, seed_input, expected, dense_model):
        activation_maximization = ActivationMaximization(dense_model)
        result = activation_maximization(CategoricalScore(0),
                                         seed_input=seed_input,
                                         input_modifiers=None,
                                         regularizers=None)
        if type(expected) is list:
            assert type(result) == list
            result = result[0]
            expected = expected[0]
        assert result.shape == expected

    @pytest.mark.parametrize("input_modifiers,expected_error", [
        (None, NO_ERROR),
        (Jitter(), ValueError),
        (Rotate(), ValueError),
        (Scale(), ValueError),
        ([], NO_ERROR),
        ([None], TypeError),
        ([Jitter()], ValueError),
        ([Rotate()], ValueError),
        ([Scale()], ValueError),
        ([Jitter(), Rotate(), Scale()], ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_input_modifiers_are_(self, input_modifiers, expected_error, dense_model):
        activation_maximization = ActivationMaximization(dense_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0),
                                             input_modifiers=input_modifiers,
                                             regularizers=None)
            assert result.shape == (1, 8)

    @pytest.mark.parametrize("regularizers,expected_error", [
        (None, NO_ERROR),
        (TotalVariation2D(), ValueError),
        (Norm(), NO_ERROR),
        ([], NO_ERROR),
        ([None], TypeError),
        ([TotalVariation2D()], ValueError),
        ([Norm()], NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision", "legacy")
    def test__call__if_regularizers_are(self, regularizers, expected_error, dense_model):
        activation_maximization = ActivationMaximization(dense_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0),
                                             input_modifiers=None,
                                             regularizers=regularizers)
            assert result.shape == (1, 8)

    @pytest.mark.parametrize("activation_modifiers,modified,expected_error", [
        (None, False, NO_ERROR),
        (lambda x: np.ones(x.shape, np.float), True, NO_ERROR),
        (dict(input_1=None), False, NO_ERROR),
        (dict(input_1=lambda x: np.ones(x.shape, np.float)), True, NO_ERROR),
        (dict(input_2=lambda x: np.ones(x.shape, np.float)), False, ValueError),
        (dict(input_1=lambda x: np.ones(x.shape, np.float),
              input_2=lambda x: np.ones(x.shape, np.float)), False, ValueError),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__with_activation_modifiers(self, activation_modifiers, modified, expected_error,
                                              dense_model):
        seed_inputs = dummy_sample((1, 8))
        activation_maximization = ActivationMaximization(dense_model)
        with assert_raises(expected_error):
            result = activation_maximization(CategoricalScore(0),
                                             seed_input=seed_inputs,
                                             input_modifiers=None,
                                             regularizers=None)
            assert not np.all(result == 1.0)
            result = activation_maximization(CategoricalScore(0),
                                             seed_input=seed_inputs,
                                             input_modifiers=None,
                                             regularizers=None,
                                             activation_modifiers=activation_modifiers)
            if modified:
                assert np.all(result == 1.0)
            else:
                assert not np.all(result == 1.0)
