from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from packaging.version import parse as version
from scipy.ndimage.interpolation import zoom

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import is_mixed_precision, standardize, zoom_factor

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import LossScaleOptimizer


class GradcamPlusPlus(Gradcam):
    """Grad-CAM++

        For details on GradCAM++, see the paper:
        [GradCAM++: Improved Visual Explanations for Deep Convolutional Networks]
        (https://arxiv.org/pdf/1710.11063.pdf).

    Todo:
        * Write examples
    """
    def __call__(self,
                 score,
                 seed_input,
                 penultimate_layer=-1,
                 seek_penultimate_conv_layer=True,
                 activation_modifier=lambda cam: K.relu(cam),
                 expand_cam=True,
                 training=False,
                 standardize_cam=True,
                 unconnected_gradients=tf.UnconnectedGradients.NONE) -> Union[np.array, list]:
        """Generate gradient based class activation maps (CAM) by using positive gradient of
            penultimate_layer with respect to score.

        Args:
            score (tf_keras_vis.utils.scores.Score|function|list):
                A function to specify visualizing target.
                If the model has multiple outputs, you can use a different
                score function on each output by passing a list of score functions.
            seed_input (tf.Tensor|np.array|list): A tensor or a list of them to input in the model.
                If the model has multiple inputs, you have to pass a list.
            penultimate_layer (int|str|tf.keras.layers.Layer, optional):
                A value to represent an index or a name of tf.keras.layers.Layer instance.
                When not None or -1, it will be the offset layer
                when seeking the penultimate `convolutional` layter.
                Defaults to None.
            seek_penultimate_conv_layer (bool, optional):
                When True to seek the penultimate `convolutional` layter that is a subtype of
                `keras.layers.convolutional.Conv` class.
                When False, `penultimate_layer` (or last layer when `penultimate_layer` is None)
                will be elected as the penultimate `convolutional` layter.
                Defaults to True.
            activation_modifier (function, optional):  A function to modify activation.
                Defaults to lambdacam:K.relu(cam).
            training (bool, optional): A bool that indicates
                whether the model's training-mode on or off.
                Defaults to False.
            expand_cam (bool, optional): True to expand cam to same as input image size.
                ![Note] When True, even if the model has multiple inputs,
                this function return only a cam value
                (That's, when `expand_cam` is True,
                multiple cam images are generated from a model that has multiple inputs).
            standardize_cam (bool, optional): When True, cam will be standardized.
                Defaults to True.
            unconnected_gradients (tf.UnconnectedGradients, optional):
                Specifies the gradient value returned when the given input tensors are unconnected.
                Defaults to tf.UnconnectedGradients.NONE.

        Returns:
            np.array|list: The class activation maps that indicate the `seed_input` regions
                whose change would most contribute the score value.

        Raises:
            ValueError: In case of invalid arguments for `score`, or `penultimate_layer`.
        """

        # Preparing
        scores = self._get_scores_for_multiple_outputs(score)
        seed_inputs = self._get_seed_inputs_for_multiple_inputs(seed_input)
        penultimate_output_tensor = self._find_penultimate_output(penultimate_layer,
                                                                  seek_penultimate_conv_layer)

        # Processing gradcam
        model = tf.keras.Model(inputs=self.model.inputs,
                               outputs=self.model.outputs + [penultimate_output_tensor])
        # When mixed precision enabled
        mixed_precision_model = is_mixed_precision(model)
        if mixed_precision_model:
            optimizer = LossScaleOptimizer(tf.keras.optimizers.RMSprop())

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(seed_inputs)
            outputs = model(seed_inputs, training=training)
            outputs, penultimate_output = outputs[:-1], outputs[-1]
            score_values = self._calculate_scores(outputs, scores)
            if mixed_precision_model:
                score_values = [
                    optimizer.get_scaled_loss(score_value) for score_value in score_values
                ]
        grads = tape.gradient(score_values,
                              penultimate_output,
                              unconnected_gradients=unconnected_gradients)

        if mixed_precision_model:
            grads = optimizer.get_unscaled_gradients(grads)
            grads = tf.cast(grads, dtype=model.variable_dtype)
            penultimate_output = tf.cast(penultimate_output, dtype=model.variable_dtype)
            score_values = [tf.cast(v, dtype=model.variable_dtype) for v in score_values]

        score = sum([tf.math.exp(tf.reshape(v, (-1, ))) for v in score_values])
        score_shape = (-1, ) + tuple(np.ones(grads.ndim - 1, np.int))
        score = tf.reshape(score, score_shape)

        first_derivative = score * grads
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
        cam = [zoom(cam, factor) for factor in factors]
        if standardize_cam:
            cam = [standardize(x) for x in cam]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cam = cam[0]
        return cam
