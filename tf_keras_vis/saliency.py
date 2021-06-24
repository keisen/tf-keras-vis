from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from . import ModelVisualization
from .utils import get_num_of_steps_allowed, listify, standardize


class Saliency(ModelVisualization):
    """Vanilla Saliency and Smooth-Grad

        For details on Vanilla Saliency, see the paper:
        [Deep Inside Convolutional Networks: Visualising Image Classification Models
        and Saliency Maps](https://arxiv.org/pdf/1312.6034)

        For details on Smooth-Grad, see the paper:
        [SmoothGrad: removing noise by adding noise](https://arxiv.org/pdf/1706.03825)

    Todo:
        * Write examples
    """
    def __call__(self,
                 score,
                 seed_input,
                 smooth_samples=0,
                 smooth_noise=0.20,
                 keepdims=False,
                 gradient_modifier=lambda grads: K.abs(grads),
                 training=False,
                 standardize_saliency=True,
                 unconnected_gradients=tf.UnconnectedGradients.NONE) -> Union[np.ndarray, list]:
        """Generate an attention map that appears how output value changes with respect to a small
            change in input image pixels.

        Args:
            score (Union[tf_keras_vis.utils.scores.Score,Callable,
                list[tf_keras_vis.utils.scores.Score,Callable]]):
                A Score instance or function to specify visualizing target. For example::

                    scores = CategoricalScore([1, 294, 413])

                This code above means the same with the one below::

                    score = lambda outputs: (outputs[0][1], outputs[1][294], outputs[2][413])

                When the model has multiple outputs, you have to pass a list of
                Score instances or functions. For example::

                    score = [
                        tf_keras_vis.utils.scores.CategoricalScore([1, 23]),  # For 1st output
                        tf_keras_vis.utils.scores.InactiveScore(),            # For 2nd output
                        ...
                    ]

            seed_input (Union[tf.Tensor,np.ndarray,list[tf.Tensor,np.ndarray]]):
                A tensor or a list of them to input in the model.
                When the model has multiple inputs, you have to pass a list.
            smooth_samples (int, optional): The number of calculating gradients iterations.
                When over zero, this method will work as SmoothGrad.
                When zero, it will work as Vanilla Saliency.
                Defaults to 0.
            smooth_noise (float, optional): Noise level. Defaults to 0.20.
            keepdims (bool, optional): A boolean that whether to keep the channels-dim or not.
                Defaults to False.
            gradient_modifier (Callable, optional): A function to modify gradients.
                Defaults to None.
            training (bool, optional): A bool that indicates
                whether the model's training-mode on or off. Defaults to False.
            standardize_saliency (bool, optional): When True, saliency map will be standardized.
                Defaults to True.
            unconnected_gradients (tf.UnconnectedGradients, optional):
                Specifies the gradient value returned when the given input tensors are unconnected.
                Defaults to tf.UnconnectedGradients.NONE.

        Returns:
            Union[np.ndarray,list]: The heatmap image indicating the `seed_input` regions
                whose change would most contribute towards maximizing the score value,
                Or a list of their images.

        Raises:
            ValueError: In case of invalid arguments for `score`, or `seed_input`.
        """

        # Preparing
        scores = self._get_scores_for_multiple_outputs(score)
        seed_inputs = self._get_seed_inputs_for_multiple_inputs(seed_input)
        # Processing saliency
        if smooth_samples > 0:
            smooth_samples = get_num_of_steps_allowed(smooth_samples)
            seed_inputs = (tf.tile(X, (smooth_samples, ) + tuple(np.ones(X.ndim - 1, np.int)))
                           for X in seed_inputs)
            seed_inputs = (tf.reshape(X, (smooth_samples, -1) + tuple(X.shape[1:]))
                           for X in seed_inputs)
            seed_inputs = ((X, tuple(range(X.ndim)[2:])) for X in seed_inputs)
            seed_inputs = ((X, smooth_noise * (tf.math.reduce_max(X, axis=axis, keepdims=True) -
                                               tf.math.reduce_min(X, axis=axis, keepdims=True)))
                           for X, axis in seed_inputs)
            seed_inputs = (X + np.random.normal(0., sigma, X.shape) for X, sigma in seed_inputs)
            seed_inputs = list(seed_inputs)
            total = (np.zeros_like(X[0]) for X in seed_inputs)
            for i in range(smooth_samples):
                grads = self._get_gradients([X[i] for X in seed_inputs], scores, gradient_modifier,
                                            training, unconnected_gradients)
                total = (total + g for total, g in zip(total, grads))
            grads = [g / smooth_samples for g in total]
        else:
            grads = self._get_gradients(seed_inputs, scores, gradient_modifier, training,
                                        unconnected_gradients)
        # Visualizing
        if not keepdims:
            grads = [np.max(g, axis=-1) for g in grads]
        if standardize_saliency:
            grads = [standardize(g) for g in grads]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            grads = grads[0]
        return grads

    def _get_gradients(self, seed_inputs, scores, gradient_modifier, training,
                       unconnected_gradients):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(seed_inputs)
            outputs = self.model(seed_inputs, training=training)
            outputs = listify(outputs)
            score_values = self._calculate_scores(outputs, scores)
        grads = tape.gradient(score_values,
                              seed_inputs,
                              unconnected_gradients=unconnected_gradients)
        if gradient_modifier is not None:
            grads = [gradient_modifier(g) for g in grads]
        return grads
