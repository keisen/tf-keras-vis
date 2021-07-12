from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.ndimage.interpolation import zoom

from . import ModelVisualization
from .utils import get_num_of_steps_allowed, is_mixed_precision, listify, standardize, zoom_factor
from .utils.model_modifiers import ExtractIntermediateLayerForGradcam as ModelModifier


class Scorecam(ModelVisualization):
    """Score-CAM and Faster Score-CAM

    References:
        * Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
          (https://arxiv.org/pdf/1910.01279.pdf)
        * Faster Score-CAM (https://github.com/tabayashi0117/Score-CAM#faster-score-cam)
    """
    def __call__(self,
                 score,
                 seed_input,
                 penultimate_layer=None,
                 seek_penultimate_conv_layer=True,
                 activation_modifier=lambda cam: K.relu(cam),
                 batch_size=32,
                 max_N=None,
                 training=False,
                 expand_cam=True,
                 standardize_cam=True) -> Union[np.ndarray, list]:
        """Generate score-weighted class activation maps (CAM) by using gradient-free
        visualization method.

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
            batch_size: The number of samples per batch. Defaults to 32.
            max_N: When None or under Zero, run as ScoreCAM. When not None and over Zero of
                Integer, run as Faster-ScoreCAM. Set larger number (or None), need more time to
                visualize CAM but to be able to get clearer attention images. Defaults to None.
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

        # Processing score-cam
        model = ModelModifier(penultimate_layer, seek_penultimate_conv_layer, False)(self.model)
        penultimate_output = model(seed_inputs, training=training)

        if is_mixed_precision(self.model):
            penultimate_output = tf.cast(penultimate_output, self.model.variable_dtype)

        # For efficiently visualizing, extract maps that has a large variance.
        # This excellent idea is devised by tabayashi0117.
        # (see for details: https://github.com/tabayashi0117/Score-CAM#faster-score-cam)
        if max_N is None or max_N <= 0:
            max_N = get_num_of_steps_allowed(penultimate_output.shape[-1])
        elif max_N > 0 and max_N <= penultimate_output.shape[-1]:
            max_N = get_num_of_steps_allowed(max_N)
        else:
            raise ValueError(f"max_N must be 1 or more and {penultimate_output.shape[-1]} or less."
                             f" max_N: {max_N}")
        if max_N < penultimate_output.shape[-1]:
            activation_map_std = tf.math.reduce_std(penultimate_output,
                                                    axis=tuple(
                                                        range(penultimate_output.ndim)[1:-1]),
                                                    keepdims=True)
            _, top_k_indices = tf.math.top_k(activation_map_std, max_N)
            top_k_indices, _ = tf.unique(tf.reshape(top_k_indices, (-1, )))
            penultimate_output = tf.gather(penultimate_output, top_k_indices, axis=-1)
        nsamples = penultimate_output.shape[0]
        channels = penultimate_output.shape[-1]

        # Upsampling activations
        input_shapes = [seed_input.shape for seed_input in seed_inputs]
        zoom_factors = (zoom_factor(penultimate_output.shape[1:-1], input_shape[1:-1])
                        for input_shape in input_shapes)
        zoom_factors = ((1, ) + factor + (1, ) for factor in zoom_factors)
        upsampled_activations = [
            zoom(penultimate_output, factor, order=1, mode='nearest') for factor in zoom_factors
        ]
        activation_shapes = [activation.shape for activation in upsampled_activations]

        # Normalizing activations
        min_activations = (np.min(activation,
                                  axis=tuple(range(activation.ndim)[1:-1]),
                                  keepdims=True) for activation in upsampled_activations)
        max_activations = (np.max(activation,
                                  axis=tuple(range(activation.ndim)[1:-1]),
                                  keepdims=True) for activation in upsampled_activations)
        normalized_activations = zip(upsampled_activations, min_activations, max_activations)
        normalized_activations = ((activation - _min) / (_max - _min + K.epsilon())
                                  for activation, _min, _max in normalized_activations)

        # (samples, h, w, c) -> (channels, samples, h, w, c)
        input_templates = (np.tile(seed_input, (channels, ) + (1, ) * len(seed_input.shape))
                           for seed_input in seed_inputs)
        # (samples, h, w, channels) -> (c, samples, h, w, channels)
        masks = (np.tile(mask, (input_shape[-1], ) + (1, ) * len(map_shape)) for mask, input_shape,
                 map_shape in zip(normalized_activations, input_shapes, activation_shapes))
        # (c, samples, h, w, channels) -> (channels, samples, h, w, c)
        masks = (np.transpose(mask, (len(mask.shape) - 1, ) + tuple(range(len(mask.shape)))[1:-1] +
                              (0, )) for mask in masks)
        # Create masked inputs
        masked_seed_inputs = (np.multiply(input_template, mask)
                              for input_template, mask in zip(input_templates, masks))
        # (channels, samples, h, w, c) -> (channels * samples, h, w, c)
        masked_seed_inputs = [
            np.reshape(seed_input, (-1, ) + seed_input.shape[2:])
            for seed_input in masked_seed_inputs
        ]

        # Predicting masked seed-inputs
        preds = self.model.predict(masked_seed_inputs, batch_size=batch_size)
        # (channels * samples, logits) -> (channels, samples, logits)
        preds = (np.reshape(prediction, (channels, nsamples, prediction.shape[-1]))
                 for prediction in listify(preds))

        # Calculating weights
        weights = ([score(K.softmax(p)) for p in prediction]
                   for score, prediction in zip(scores, preds))
        weights = ([self._validate_weight(s, nsamples) for s in w] for w in weights)
        weights = (np.array(w, dtype=np.float32) for w in weights)
        weights = (np.reshape(w, (channels, nsamples, -1)) for w in weights)
        weights = (np.mean(w, axis=2) for w in weights)
        weights = (np.transpose(w, (1, 0)) for w in weights)
        weights = np.array(list(weights), dtype=np.float32)
        weights = np.sum(weights, axis=0)

        # Generate cam
        cam = K.batch_dot(penultimate_output, weights)
        if activation_modifier is not None:
            cam = activation_modifier(cam)

        if not expand_cam:
            if standardize_cam:
                cam = standardize(cam)
            return cam

        # Visualizing
        zoom_factors = (zoom_factor(cam.shape, X.shape) for X in seed_inputs)
        cam = [zoom(cam, factor, order=1) for factor in zoom_factors]
        if standardize_cam:
            cam = [standardize(x) for x in cam]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            cam = cam[0]
        return cam

    def _validate_weight(self, score, nsamples):
        invalid = False
        if tf.is_tensor(score) or isinstance(score, np.ndarray):
            invalid = (score.shape[0] != nsamples)
        elif isinstance(score, (list, tuple)):
            invalid = (len(score) != nsamples)
        else:
            invalid = (nsamples != 1)
        if invalid:
            raise ValueError(
                "Score function must return a Tensor, whose the first dimension is "
                "the same as the first dimension of seed_input or "
                ", a list or tuple, whose length is the first dimension of seed_input.")
        else:
            return score


ScoreCAM = Scorecam
