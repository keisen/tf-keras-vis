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
                 gradient_modifier=lambda grads: K.abs(grads)):
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
            keepdims: A boolean that whether to keep the channels-dim or not.
            smooth_noise: Noise level that is recommended no tweaking when there is no reason.
            gradient_modifier: A function to modify gradients. By default, the function modify
                gradients to `absolute` values.
        # Returns
            The heatmap image indicating the `seed_input` regions whose change would most contribute
            towards maximizing the loss value, Or a list of their images.
            A list of Numpy arrays that the model inputs that maximize the out of `loss`.
        # Raises
            ValueError: In case of invalid arguments for `loss`, or `seed_input`.
        """
        losses = self._prepare_losses(loss)
        seed_inputs = [X if tf.is_tensor(X) else tf.constant(X) for X in listify(seed_input)]
        seed_inputs = [
            tf.expand_dims(seed_input, axis=0) if X.shape == input_tensor.shape[1:] else X
            for X, input_tensor in zip(seed_inputs, self.model.inputs)
        ]
        if len(seed_inputs) != len(self.model.inputs):
            raise ValueError('')

        if smooth_samples > 0:
            axes = [tuple(range(1, len(X.shape))) for X in seed_inputs]
            sigmas = [
                smooth_noise * (np.max(X, axis=axis) - np.min(X, axis=axis))
                for X, axis in zip(seed_inputs, axes)
            ]
            total_gradients = (np.zeros_like(X) for X in seed_inputs)
            for i in range(check_steps(smooth_samples)):
                seed_inputs_plus_noise = [
                    tf.constant(
                        np.concatenate([
                            x + np.random.normal(0., s, (1, ) + x.shape) for x, s in zip(X, sigma)
                        ])) for X, sigma in zip(seed_inputs, sigmas)
                ]
                gradients, loss_values, outputs = self._get_gradients(self.model,
                                                                      seed_inputs_plus_noise,
                                                                      losses, gradient_modifier)
                total_gradients = (total + g for total, g in zip(total_gradients, gradients))
            grads = [g / smooth_samples for g in total_gradients]
        else:
            grads, loss_values, outputs = self._get_gradients(self.model, seed_inputs, losses,
                                                              gradient_modifier)

        if not keepdims:
            grads = [np.max(g, axis=-1) for g in grads]
        if len(self.model.inputs) == 1 and not isinstance(seed_input, list):
            grads = grads[0]
        return grads

    def _get_gradients(self, model, seed_inputs, losses, gradient_modifier):
        with tf.GradientTape() as tape:
            tape.watch(seed_inputs)
            outputs = model(seed_inputs)
            outputs = listify(outputs)
            loss_values = [loss(output) for output, loss in zip(outputs, losses)]
        grads = tape.gradient(loss_values, seed_inputs)
        if gradient_modifier is not None:
            grads = [gradient_modifier(g) for g in grads]
        return grads, loss_values, outputs
