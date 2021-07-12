from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.ndimage.interpolation import zoom

from . import ModelVisualization
from .utils import is_mixed_precision, standardize, zoom_factor
from .utils.model_modifiers import ExtractIntermediateLayerForGradcam as ModelModifier


class GradcamPlusPlus(ModelVisualization):
    """Grad-CAM++

    References:
        * GradCAM++: Improved Visual Explanations for Deep Convolutional Networks
          (https://arxiv.org/pdf/1710.11063.pdf)
    """
    def __call__(self,
                 score,
                 seed_input,
                 penultimate_layer=None,
                 seek_penultimate_conv_layer=True,
                 activation_modifier=lambda cam: K.relu(cam),
                 training=False,
                 expand_cam=True,
                 standardize_cam=True,
                 unconnected_gradients=tf.UnconnectedGradients.NONE) -> Union[np.ndarray, list]:
        """Generate gradient based class activation maps (CAM) by using positive gradient of
            penultimate_layer with respect to score.

        Args:
            score: A :obj:`tf_keras_vis.utils.scores.Score` instance, function or a list of them.
                For example of the Score instance to specify visualizing target::

                    scores = CategoricalScore([1, 294, 413])

                The code above means the same with the one below::

                    score = lambda outputs: (outputs[0][1], outputs[1][294], outputs[2][413])

                When the model has multiple outputs, you MUST pass a list of
                Score instances or functions. For example::

                    from tf_keras_vis.utils.scores import CategoricalScore, InactiveScore
                    score = [
                        CategoricalScore([1, 23]),  # For 1st model output
                        InactiveScore(),            # For 2nd model output
                        ...
                    ]

            seed_input: A tf.Tensor, :obj:`numpy.ndarray` or a list of them to input in the model.
                That's when the model has multiple inputs, you MUST pass a list of tensors.
            penultimate_layer: An index or name of the layer, or the tf.keras.layers.Layer
                instance itself. When None, it means the same with `-1`. If the layer specified by
                this option is not `convolutional` layer, `penultimate_layer` will work as the
                offset to seek `convolutional` layer. Defaults to None.
            seek_penultimate_conv_layer: A bool that indicates whether or not seeks a penultimate
                layer when the layer specified by `penultimate_layer` is not `convolutional` layer.
                Defaults to True.
            activation_modifier: A function to modify the Class Activation Map (CAM). Defaults to
                `lambda cam: K.relu(cam)`.
            training: A bool that indicates whether the model's training-mode on or off. Defaults
                to False.
            expand_cam: True to resize CAM to the same as input image size. **Notes!** When False,
                even if the model has multiple inputs, return only a CAM. Defaults to True.
            standardize_cam: When True, CAM will be standardized. Defaults to True.
            unconnected_gradients: Specifies the gradient value returned when the given input
                tensors are unconnected. Defaults to tf.UnconnectedGradients.NONE.

        Returns:
            An :obj:`numpy.ndarray` or a list of them. They are the Class Activation Maps (CAMs)
            that indicate the `seed_input` regions whose change would most contribute the score
            value.

        Raises:
            :obj:`ValueError`: When there is any invalid arguments.
        """

        # Preparing
        scores = self._get_scores_for_multiple_outputs(score)
        seed_inputs = self._get_seed_inputs_for_multiple_inputs(seed_input)

        # Processing gradcam
        model = ModelModifier(penultimate_layer, seek_penultimate_conv_layer)(self.model)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(seed_inputs)
            outputs = model(seed_inputs, training=training)
            outputs, penultimate_output = outputs[:-1], outputs[-1]
            score_values = self._calculate_scores(outputs, scores)
        grads = tape.gradient(score_values,
                              penultimate_output,
                              unconnected_gradients=unconnected_gradients)

        # When mixed precision enabled
        if is_mixed_precision(model):
            grads = tf.cast(grads, dtype=model.variable_dtype)
            penultimate_output = tf.cast(penultimate_output, dtype=model.variable_dtype)
            score_values = [tf.cast(v, dtype=model.variable_dtype) for v in score_values]

        score_values = sum(tf.math.exp(o) for o in score_values)
        score_values = tf.reshape(score_values, score_values.shape + (1, ) * (grads.ndim - 1))

        first_derivative = score_values * grads
        second_derivative = first_derivative * grads
        third_derivative = second_derivative * grads

        global_sum = K.sum(penultimate_output,
                           axis=tuple(np.arange(len(penultimate_output.shape))[1:-1]),
                           keepdims=True)

        alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
        alpha_denom = alpha_denom + tf.cast((second_derivative == 0.0), second_derivative.dtype)
        alphas = second_derivative / alpha_denom

        alpha_normalization_constant = K.sum(alphas,
                                             axis=tuple(np.arange(len(alphas.shape))[1:-1]),
                                             keepdims=True)
        alpha_normalization_constant = alpha_normalization_constant + tf.cast(
            (alpha_normalization_constant == 0.0), alpha_normalization_constant.dtype)
        alphas = alphas / alpha_normalization_constant

        if activation_modifier is None:
            weights = first_derivative
        else:
            weights = activation_modifier(first_derivative)
        deep_linearization_weights = weights * alphas
        deep_linearization_weights = K.sum(
            deep_linearization_weights,
            axis=tuple(np.arange(len(deep_linearization_weights.shape))[1:-1]),
            keepdims=True)

        cam = K.sum(deep_linearization_weights * penultimate_output, axis=-1)
        if activation_modifier is not None:
            cam = activation_modifier(cam)

        if not expand_cam:
            if standardize_cam:
                cam = standardize(cam)
            return cam

        # Visualizing
        factors = (zoom_factor(cam.shape, X.shape) for X in seed_inputs)
        cam = [zoom(cam, factor, order=1) for factor in factors]
        if standardize_cam:
            cam = [standardize(x) for x in cam]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cam = cam[0]
        return cam
