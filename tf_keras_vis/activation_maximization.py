import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tf_keras_vis import ModelVisualization
from tf_keras_vis.utils import check_steps, listify
from tf_keras_vis.utils.input_modifiers import Jitter, Rotate
from tf_keras_vis.utils.regularizers import L2Norm, TotalVariation


class ActivationMaximization(ModelVisualization):
    def __call__(self,
                 loss,
                 seed_input=None,
                 input_range=(0, 255),
                 input_modifiers=[Jitter(2), Rotate(2)],
                 regularizers=[TotalVariation(10.), L2Norm(10.)],
                 steps=200,
                 optimizer=tf.optimizers.RMSprop(1., 0.95),
                 normalize_gradient=True,
                 gradient_modifier=None,
                 callbacks=None):
        """Generate the model inputs that maximize the output of the given `loss` functions.

        # Arguments
            loss: A loss function. If the model has multipul outputs, you can use a different
                loss on each output by passing a list of losses. The loss value that will be
                maximized will then be the sum of all individual losses
                (and all regularization values).
            seed_input: `None`(default) or An N-dim Numpy array. When `None`, the seed_input
                value will be generated with randome noise.  If the model has multipul inputs,
                you have to pass a list of N-dim Numpy arrays.
            input_range: A tuple that specifies the input range as a `(min, max)` tuple. If the
                model has multipul inputs, you can use a different input range on each input by
                passing as list of input ranges. When `None` or a `(None, None)` tuple, the range of
                a input value (i.e., the result of this function) will be no applied any limitation.
            input_modifiers: A input modifier function or a list of input modifier functions.
                You can also use a instance of `tf_keras-vis.input_modifiers. InputModifier`'s
                subclass, instead of a function. If the model has multipul inputs, you have to pass
                a dictionary of input modifier functions or instances on each model inputs:
                such as `input_modifiers={'input_a': [ input_modifier_a_1, input_modifier_a_2 ],
                'input_b': input_modifier_b, ... }`.
            regularizers: A regularization function or a list of regularization functions. You can
                also use a instance of `tf_keras-vis.regularizers.Regularizer`'s subclass,
                instead of a function. If the model has multipul outputs, you can use
                a dictionary of regularization functions or instances on each model outputs:
                such as `regularizers={'output_a': [ regularizer_a_1, regularizer_a_2 ],
                'output_b': regularizer_b, ... }`. A regularization value will be calculated with
                a corresponding model input will add to the loss value.
            steps: The number of gradient descent iterations.
            optimizer: A `tf.optimizers.Optimizer` instance.
            normalize_gradient: True to normalize gradients. Normalization avoids too small or
                large gradients and ensures a smooth gradient descent process.
            gradient_modifier: A function to modify gradients. This function is executed before
                normalizing gradients.
            callbacks: A `tf_keras_vis.callbacks.OptimizerCallback` instance or a list of them.
        # Returns
            An Numpy arrays when the model has a single input and `seed_input` is None or An N-dim
            Numpy Array, Or a list of Numpy arrays when otherwise. The Numpy is that the model
            inputs that maximize the out of `loss`.
        # Raises
            ValueError: In case of invalid arguments for `loss`, `input_range`, `input_modifiers`
                or `regularizers`.
        """
        # losses
        losses = self._prepare_losses(loss)

        # Get initial seed-inputs
        input_ranges = self._prepare_input_ranges(input_range)
        seed_inputs = self._get_seed_inputs(seed_input, input_ranges)

        # input_modifiers
        input_modifiers = self._prepare_inputmodifier_dictionary(input_modifiers)

        # regularizers
        regularizers = self._prepare_regularizer_dictionary(regularizers)

        callbacks = listify(callbacks)
        for callback in callbacks:
            callback.on_begin()

        for i in range(check_steps(steps)):
            # Apply input modifiers
            for j, input_layer in enumerate(self.model.inputs):
                for modifier in input_modifiers[input_layer.name]:
                    seed_inputs[j] = modifier(seed_inputs[j])
            seed_inputs = [tf.Variable(X) for X in seed_inputs]

            # Calculate gradients
            with tf.GradientTape() as tape:
                tape.watch(seed_inputs)
                outputs = self.model(seed_inputs)
                outputs = listify(outputs)
                loss_values = [loss(output) for output, loss in zip(outputs, losses)]
                # Calculate regularization values
                regularization_values = [[(regularizer.name, regularizer(seed_inputs))
                                          for regularizer in regularizers[output_layer.name]]
                                         for output_layer in self.model.outputs]
                ys = [
                    (-1. * loss_value) + sum([rv for (_, rv) in regularization_value])
                    for loss_value, regularization_value in zip(loss_values, regularization_values)
                ]
            grads = tape.gradient(ys, seed_inputs)
            grads = listify(grads)
            if gradient_modifier is not None:
                grads = [gradient_modifier(g) for g in grads]
            if normalize_gradient:
                grads = [K.l2_normalize(g) for g in grads]
            optimizer.apply_gradients(zip(grads, seed_inputs))

            for callback in callbacks:
                callback(i,
                         self._apply_clip(seed_inputs, input_ranges),
                         grads,
                         loss_values,
                         outputs,
                         regularizations=regularization_values,
                         overall_loss=ys)

        for callback in callbacks:
            callback.on_end()

        cliped_value = self._apply_clip(seed_inputs, input_ranges)
        if len(self.model.inputs) == 1 and (seed_input is None or not isinstance(seed_input, list)):
            cliped_value = cliped_value[0]

        return cliped_value

    def _prepare_input_ranges(self, input_range):
        model_inputs_length = len(self.model.inputs)
        input_ranges = listify(input_range, empty_list_if_none=False, convert_tuple_to_list=False)
        if len(input_ranges) == 1 and model_inputs_length > 1:
            input_ranges = input_ranges * model_inputs_length
        """
        if len(input_ranges) < model_inputs_length:
            input_ranges = input_ranges + [None] * model_inputs_length - len(input_ranges)
        """
        input_ranges = [(None, None) if r is None else r for r in input_ranges]
        for i, r in enumerate(input_ranges):
            if len(r) != 2:
                raise ValueError(
                    'the length of input rage tuple must be 2 (Or it is just `None`, not tuple), '
                    'but you passed {} as `input_ranges[{}]`.'.format(r, i))
        return input_ranges

    def _get_seed_inputs(self, seed_inputs, input_ranges):
        # Prepare input_ranges
        input_ranges = ((low, 1.) if high is None else (low, high) for (low, high) in input_ranges)
        input_ranges = ((0., high) if low is None else (low, high) for (low, high) in input_ranges)
        input_ranges = ((low, high, (high - low) * 0.1) for (low, high) in input_ranges)
        # Prepare input_shape
        input_shapes = (input_tensor.shape[1:] for input_tensor in self.model.inputs)
        # Prepare seed_inputs
        if seed_inputs is None or len(seed_inputs) == 0:
            seed_inputs = [None] * len(self.model.inputs)
            seed_inputs = (tf.random.uniform(shape, low + margin, high - margin)
                           for X, (low, high,
                                   margin), shape in zip(seed_inputs, input_ranges, input_shapes))
        else:
            seed_inputs = listify(seed_inputs)
        # replace numpy array to tf-tensor
        seed_inputs = (X if tf.is_tensor(X) else tf.Variable(X, dtype=input_tensor.dtype)
                       for X, input_tensor in zip(seed_inputs, self.model.inputs))
        # add dimension when tensor doesn't have the dim for samples
        seed_inputs = (tf.expand_dims(X, axis=0) if len(X.shape) < len(input_tensor.shape) else X
                       for X, input_tensor in zip(seed_inputs, self.model.inputs))
        return list(seed_inputs)

    def _prepare_inputmodifier_dictionary(self, input_modifier):
        input_modifiers = listify(input_modifier)
        input_modifiers = self._prepare_dictionary(input_modifiers,
                                                   [l.name for l in self.model.inputs])
        if len(input_modifiers) != 0 and len(input_modifiers) != len(self.model.inputs):
            raise ValueError('The model has {} inputs, but you passed {} as input_modifiers. '
                             'When the model has multipul inputs, '
                             'you must pass a dictionary as input_modifiers.'.format(
                                 len(self.model.inputs), input_modifier))
        return input_modifiers

    def _prepare_regularizer_dictionary(self, regularizer):
        regularizers = self._prepare_dictionary(regularizer, [l.name for l in self.model.outputs])
        if len(regularizers) != len(self.model.outputs):
            raise ValueError('The model has {} outputs, but you passed {} as regularizers. '
                             'When the model has multipul outputs, '
                             'you must pass a dictionary as regularizers.'.format(
                                 len(self.model.outputs), regularizers))
        return regularizers

    def _apply_clip(self, seed_inputs, input_ranges):
        return [np.array(K.clip(X, low, high)) for X, (low, high) in zip(seed_inputs, input_ranges)]
