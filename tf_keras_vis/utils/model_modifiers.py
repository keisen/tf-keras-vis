from abc import ABC, abstractmethod
from typing import Union

import tensorflow as tf
from packaging.version import parse as version

if version(tf.version.VERSION) < version("2.6.0rc0"):
    from tensorflow.python.keras.layers.convolutional import Conv
else:
    from keras.layers.convolutional import Conv

from . import find_layer


class ModelModifier(ABC):
    """Abstract class for defining a model modifier.
    """
    @abstractmethod
    def __call__(self, model) -> Union[None, tf.keras.Model]:
        """Implement modification to the model before processing gradient descent.

        Args:
            model: A model instance.

        Raises:
            NotImplementedError: This method must be overwritten.

        Returns: Modified model or None.
        """
        raise NotImplementedError()


class ReplaceToLinear(ModelModifier):
    """A model modifier that replaces the activation functions of all output layers to
    `tf.keras.activations.linear`.

    Please note that this modifier must be set the end of modifiers list
    that is passed to `ModelVisualization#__init__()`. For example::

        # When visualizing `intermediate-1` layer.
        ActivationMaximization(YOUR_MODEL,
                                model_modifier=[ExtractIntermediateLayer("intermediate-1"),
                                                ReplaceToLinear()])
    """
    def __call__(self, model) -> None:
        layers = (model.get_layer(name=name) for name in model.output_names)
        for layer in layers:
            layer.activation = tf.keras.activations.linear


class ExtractIntermediateLayer(ModelModifier):
    """A model modifier that constructs new model instance
    whose output layer is an intermediate layer of `model`.

    This modifier will be used to visualize the features of the model layer.
    """
    def __init__(self, index_or_name) -> None:
        if not isinstance(index_or_name, (str, int)):
            raise TypeError("The type of `index_or_name` must be a object of string or integer."
                            f"index_or_name: {index_or_name}")
        self.index_or_name = index_or_name

    def __call__(self, model) -> tf.keras.Model:
        if isinstance(self.index_or_name, int):
            target_layer = model.get_layer(index=self.index_or_name)
        if isinstance(self.index_or_name, str):
            target_layer = model.get_layer(name=self.index_or_name)
        return tf.keras.Model(inputs=model.inputs, outputs=target_layer.output)


class GuidedBackpropagation(ModelModifier):
    """A model modifier that replaces the gradient calculation of activation functions to
    Guided calculation.

    For details on Guided back propagation, see the papers:

    References:
        * String For Simplicity: The All Convolutional Net (https://arxiv.org/pdf/1412.6806.pdf)
        * Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via
          Gradient-based Localization (https://arxiv.org/pdf/1610.02391v1.pdf)

    Warnings:
        Please note that there is a discussion that Guided Backpropagation is not working well as
        model explanations.

        * Sanity Checks for Saliency Maps (https://arxiv.org/pdf/1810.03292.pdf)
        * Guided Grad-CAM is Broken! Sanity Checks for Saliency Maps
          (https://glassboxmedicine.com/2019/10/12/guided-grad-cam-is-broken-sanity-checks-for-saliency-maps/)
    """
    def __init__(self, target_activations=[tf.keras.activations.relu]) -> None:
        self.target_activations = target_activations

    def _get_guided_activation(self, activation):
        @tf.custom_gradient
        def guided_activation(x):
            def grad(dy):
                return tf.cast(dy > 0, dy.dtype) * tf.cast(x > 0, dy.dtype) * dy

            return activation(x), grad

        return guided_activation

    def __call__(self, model) -> None:
        for layer in (layer for layer in model.layers if hasattr(layer, "activation")):
            if layer.activation in self.target_activations:
                layer.activation = self._get_guided_activation(layer.activation)


class ExtractIntermediateLayerForGradcam(ModelModifier):
    def __init__(self, penultimate_layer=None, seek_conv_layer=True, include_model_outputs=True):
        self.penultimate_layer = penultimate_layer
        self.seek_conv_layer = seek_conv_layer
        self.include_model_outputs = include_model_outputs

    def __call__(self, model):
        _layer = self.penultimate_layer
        if not isinstance(_layer, tf.keras.layers.Layer):
            if _layer is None:
                _layer = -1
            if isinstance(_layer, int) and _layer < len(model.layers):
                _layer = model.layers[_layer]
            elif isinstance(_layer, str):
                _layer = find_layer(model, lambda l: l.name == _layer)
            else:
                raise ValueError(f"Invalid argument. `penultimate_layer`={self.penultimate_layer}")
        if _layer is not None and self.seek_conv_layer:
            _layer = find_layer(model, lambda l: isinstance(l, Conv), offset=_layer)
        if _layer is None:
            raise ValueError("Unable to determine penultimate `Conv` layer. "
                             f"`penultimate_layer`={self.penultimate_layer}")
        penultimate_output = _layer.output
        if len(penultimate_output.shape) < 3:
            raise ValueError(
                "Penultimate layer's output tensor MUST have "
                f"samples, spaces and channels dimensions. [{penultimate_output.shape}]")
        outputs = [penultimate_output]
        if self.include_model_outputs:
            outputs = model.outputs + outputs
        return tf.keras.Model(inputs=model.inputs, outputs=outputs)
