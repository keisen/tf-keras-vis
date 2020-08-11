import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tf_keras_vis import ModelVisualization
from tf_keras_vis.utils import check_steps, listify


class Saliency(ModelVisualization):
    def __call__(self,
                 loss,
                 seed_input,
                 smooth_samples=0,
                 smooth_noise=0.20,
                 keepdims=False,
                 gradient_modifier=lambda grads: K.abs(grads),
                 training=False):
        """Generate an attention map that appears how output value changes with respect to a small
            change in input image pixels.
            See details: https://arxiv.org/pdf/1706.03825.pdf

        # Arguments
            loss: A loss function. If the model has multiple outputs, you can use a different
                loss on each output by passing a list of losses.
            seed_input: An N-dim Numpy array. If the model has multiple inputs,
                you have to pass a list of N-dim Numpy arrays.
            smooth_samples: The number of calculating gradients iterations. If set to zero,
                the noise for smoothing won't be generated.
            smooth_noise: Noise level that is recommended no tweaking when there is no reason.
            keepdims: A boolean that whether to keep the channels-dim or not.
            gradient_modifier: A function to modify gradients. By default, the function modify
                gradients to `absolute` values.
            training: A bool whether the model's trainig-mode turn on or off.
        # Returns
            The heatmap image indicating the `seed_input` regions whose change would most contribute
            towards maximizing the loss value, Or a list of their images.
            A list of Numpy arrays that the model inputs that maximize the out of `loss`.
        # Raises
            ValueError: In case of invalid arguments for `loss`, or `seed_input`.
        """
        # Preparing
        losses = self._get_losses_for_multiple_outputs(loss)
        seed_inputs = self._get_seed_inputs_for_multiple_inputs(seed_input)
        # Processing saliency
        if smooth_samples > 0:
            smooth_samples = check_steps(smooth_samples)
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
                grads = self._get_gradients([X[i] for X in seed_inputs], losses, gradient_modifier,
                                            training)
                total = (total + g for total, g in zip(total, grads))
            grads = [g / smooth_samples for g in total]
        else:
            grads = self._get_gradients(seed_inputs, losses, gradient_modifier, training)
        # Visualizing
        if not keepdims:
            grads = [np.max(g, axis=-1) for g in grads]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            grads = grads[0]
        return grads

    def _get_gradients(self, seed_inputs, losses, gradient_modifier, training):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(seed_inputs)
            outputs = self.model(seed_inputs, training=training)
            outputs = listify(outputs)
            loss_values = [loss(output) for output, loss in zip(outputs, losses)]
        grads = tape.gradient(loss_values,
                              seed_inputs,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
        if gradient_modifier is not None:
            grads = [gradient_modifier(g) for g in grads]
        return grads
