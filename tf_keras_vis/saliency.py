import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tf_keras_vis import ModelVisualization
from tf_keras_vis.utils import listify


class Saliency(ModelVisualization):
    def __init__(self, model, model_modifier=None):
        """Create an Saliency that support SmoothGrads.

        # Arguments
            model: The `tf.keras.Model` instance. This model will be cloned by
                `tf.keras.models.clone_model` function and then will be modified by `model_modifier`
                according to need. Therefore the model will be NOT modified.
            model_modifier: A function that modify `model` instance.
        """
        super().__init__(model, model_modifier=model_modifier)

    def __call__(self,
                 loss,
                 seed_input,
                 smooth_samples=0,
                 smooth_noise=0.15,
                 gradient_modifier=lambda grads: K.abs(grads),
                 callbacks=None):
        """Generate an attention map that appears how output value changes with respect to a small
            change in input image pixels.
            See details: https://arxiv.org/pdf/1706.03825.pdf

        # Arguments
            loss: A loss function. If the model has multipul outputs, you can use a different
                loss on each output by passing a list of losses.
            seed_input: An N-dim Numpy array. If the model has multipul inputs,
                you have to pass a list of N-dim Numpy arrays.
            smooth_samples: The number of calculating gradients iterations. If set to zero,
                the noise for smoothing won't be generated.
            smooth_noise: Noise level that is recommended no tweaking when there is no reason.
            gradient_modifier: A function to modify gradients. By default, the function modify
                gradients to `absolute` values.
            callbacks: A `tf_keras_vis.callbacks.Callback` instance or a list of them.
        # Returns
            The heatmap image indicating the `seed_input` regions whose change would most contribute
            towards maximizing the loss value, Or a list of their images.
            A list of Numpy arrays that the model inputs that maximize the out of `loss`.
        # Raises
            ValueError: In case of invalid arguments for `loss`.
        """
        losses = self._prepare_losses(loss)
        seed_inputs = [X if tf.is_tensor(X) else tf.constant(X) for X in listify(seed_input)]
        callbacks = listify(callbacks)

        if smooth_samples > 0:
            axes = [tuple(range(1, len(X.shape))) for X in seed_inputs]
            sigmas = [
                smooth_noise * (np.max(X, axis=axis) - np.min(X, axis=axis))
                for X, axis in zip(seed_inputs, axes)
            ]
            total_gradients = [np.zeros_like(X) for X in seed_inputs]
            for i in range(smooth_samples):
                seed_inputs_plus_noise = [
                    tf.constant(
                        np.concatenate([
                            x + np.random.normal(0., s, (1, ) + x.shape) for x, s in zip(X, sigma)
                        ])) for X, sigma in zip(seed_inputs, sigmas)
                ]
                gradients = self._get_gradients(self.model, seed_inputs_plus_noise, losses)
                total_gradients = [total + g for total, g in zip(total_gradients, gradients)]
                for c in callbacks:
                    grads = [g / smooth_samples for g in total_gradients]
                    c(i, self._post_process(grads, gradient_modifier, seed_input))
            grads = [g / smooth_samples for g in total_gradients]
        else:
            grads = self._get_gradients(self.model, seed_inputs, losses)
        return self._post_process(grads, gradient_modifier, seed_input)

    def _get_gradients(self, model, seed_inputs, losses):
        with tf.GradientTape() as tape:
            tape.watch(seed_inputs)
            outputs = model(seed_inputs)
            outputs = listify(outputs)
            loss_values = [loss(output) for output, loss in zip(outputs, losses)]
        return tape.gradient(loss_values, seed_inputs)

    def _post_process(self, grads, gradient_modifier, seed_input):
        if gradient_modifier is not None:
            grads = [gradient_modifier(g) for g in grads]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            grads = grads[0]
        return grads
