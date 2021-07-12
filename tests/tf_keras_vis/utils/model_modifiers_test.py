import pytest
import tensorflow as tf

from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import (ExtractIntermediateLayer, GuidedBackpropagation,
                                                ReplaceToLinear)
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.test import (NO_ERROR, assert_raises, dummy_sample, mock_conv_model,
                                     mock_multiple_outputs_model)


class TestReplaceToLinear():
    @pytest.mark.parametrize("model", [mock_conv_model(), mock_multiple_outputs_model()])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__(self, model):
        assert model.get_layer(name='output_1').activation != tf.keras.activations.linear
        if len(model.outputs) > 1:
            assert model.get_layer(name='output_2').activation != tf.keras.activations.linear
        instance = ActivationMaximization(model, model_modifier=ReplaceToLinear())
        assert instance.model != model
        assert instance.model.get_layer(name='output_1').activation == tf.keras.activations.linear
        if len(model.outputs) > 1:
            assert instance.model.get_layer(
                name='output_2').activation == tf.keras.activations.linear
            instance([CategoricalScore(0), CategoricalScore(0)])
        else:
            instance([CategoricalScore(0)])


class TestExtractIntermediateLayer():
    @pytest.mark.parametrize("model", [mock_conv_model(), mock_multiple_outputs_model()])
    @pytest.mark.parametrize("layer,expected_error", [
        (None, TypeError),
        (1, NO_ERROR),
        ('conv_1', NO_ERROR),
    ])
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__(self, model, layer, expected_error):
        assert model.outputs[0].shape.as_list() == [None, 2]
        with assert_raises(expected_error):
            instance = ActivationMaximization(model,
                                              model_modifier=ExtractIntermediateLayer(layer))
            assert instance.model != model
            assert instance.model.outputs[0].shape.as_list() == [None, 6, 6, 6]
            instance([CategoricalScore(0)])


class TestExtractIntermediateLayerForGradcam():
    pass


class TestExtractGuidedBackpropagation():
    @pytest.mark.usefixtures("mixed_precision")
    def test__call__(self, conv_model):
        instance = Saliency(conv_model, model_modifier=GuidedBackpropagation())
        guided_model = instance.model
        assert guided_model != conv_model
        assert guided_model.get_layer('conv_1').activation != conv_model.get_layer(
            'conv_1').activation
        assert guided_model.get_layer('dense_1').activation == conv_model.get_layer(
            'dense_1').activation
        instance(CategoricalScore(0), dummy_sample((1, 8, 8, 3)))
