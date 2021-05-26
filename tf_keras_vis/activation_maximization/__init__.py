import warnings
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from packaging.version import parse as version

from tf_keras_vis import ModelVisualization
from tf_keras_vis.utils import (check_steps, is_mixed_precision, listify, lower_precision_dtype)
from tf_keras_vis.utils.input_modifiers import Jitter, Rotate
from tf_keras_vis.utils.regularizers import Norm, TotalVariation2D

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import LossScaleOptimizer


class ActivationMaximization(ModelVisualization):
    def __call__(
            self,
            score,
            seed_input=None,
            input_range=(0, 255),
            input_modifiers=[Jitter(jitter=8), Rotate(degree=3)],
            regularizers=[TotalVariation2D(weight=1.),
                          Norm(weight=1., p=2)],
            steps=200,
            optimizer=None,  # When None, the default is tf.optimizers.RMSprop(1.0, 0.95)
            normalize_gradient=None,  # Disabled option.
            gradient_modifier=None,
            callbacks=None,
            training=False,
            unconnected_gradients=tf.UnconnectedGradients.NONE):
        """Generate the model inputs that maximize the output of the given `score` functions.

        # Arguments
            score: A score function or a list of score functions.
                If the model has multiple outputs, you can use a different function
                on each output by passing a list of functions. The score value that will be
                maximized will be the mean of all individual score functions
                (and sum of all regularization values).
            seed_input: `None`(default), an N-dim Numpy array or a list of N-dim Numpy arrays.
                When `None`, the seed_input value will be generated with randome uniform noise.
                If the model has multiple inputs, you have to pass a list of N-dim Numpy arrays.
            input_range: A tuple that specifies the input range as a `(min, max)` tuple
                or a list of the tuple. If the model has multiple inputs, you can use
                a different input range on each input by passing as list of input ranges.
                When `None` or a `(None, None)` tuple, the range of a input value
                (i.e., the result of this function) will be no applied any limitation.
            input_modifiers: A input modifier function, a list of input modifier functions,
                or a dictionary that has a list of input_modifiers functions.
                You can also use a instance of `tf_keras-vis.utils.input_modifiers.InputModifier`'s
                subclass, instead of a function. If the model has multiple inputs, you have to pass
                a dictionary of input modifier functions or instances on each model inputs:
                such as `input_modifiers={'input_a': [ input_modifier_a_1, input_modifier_a_2 ],
                'input_b': input_modifier_b, ... }`.
            regularizers: A regularization function or a list of regularization functions. You can
                also use a instance of `tf_keras-vis.utils.regularizers.Regularizer`'s subclass,
                instead of a function. A regularization value will be calculated with
                a corresponding model input will add to the score value.
            steps: The number of gradient descent iterations.
            optimizer: A `tf.optimizers.Optimizer` instance.
            normalize_gradient: Note! This option is now disabled.
            gradient_modifier: A function to modify gradients. This function is executed before
                normalizing gradients.
            callbacks: A `tf_keras_vis.activation_maimization.callbacks.Callback` instance
                or a list of them.
            training: A bool whether the model's trainig-mode turn on or off.
            unconnected_gradients: Specifies the gradient value returned when the given input
                tensors are unconnected. Accepted values are constants defined in the class
                `tf.UnconnectedGradients` and the default value is NONE.
        # Returns
            An Numpy arrays when the model has a single input and `seed_input` is None or An N-dim
            Numpy Array, Or a list of Numpy arrays when otherwise.
        # Raises
            ValueError: In case of invalid arguments for `score`, `input_range`, `input_modifiers`
                or `regularizers`.
        """
        if normalize_gradient is not None:
            warnings.warn(
                ('`normalize_gradient` option of ActivationMaximization#__call__() is disabled.,'
                 ' And this will be removed in future.'), DeprecationWarning)

        # Check model
        mixed_precision_model = is_mixed_precision(self.model)

        # optimizer
        optimizer = self._get_optimizer(optimizer, mixed_precision_model)

        # scores
        scores = self._get_scores_for_multiple_outputs(score)

        # Get initial seed-inputs
        input_ranges = self._get_input_ranges(input_range)
        seed_inputs = self._get_seed_inputs(seed_input, input_ranges)

        # input_modifiers
        input_modifiers = self._get_input_modifiers(input_modifiers)

        # regularizers
        regularizers = self._get_regularizers(regularizers)

        callbacks = listify(callbacks)
        for callback in callbacks:
            callback.on_begin()

        for i in range(check_steps(steps)):
            # Apply input modifiers
            for j, name in enumerate(self.model.input_names):
                for modifier in input_modifiers[name]:
                    seed_inputs[j] = modifier(seed_inputs[j])

            if mixed_precision_model:
                seed_inputs = (tf.cast(X, dtype=lower_precision_dtype(self.model))
                               for X in seed_inputs)
            seed_inputs = [tf.Variable(X) for X in seed_inputs]

            # Calculate gradients
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(seed_inputs)
                outputs = self.model(seed_inputs, training=training)
                outputs = listify(outputs)
                score_values = self._calculate_scores(outputs, scores)
                # Calculate regularization values
                regularizations = [(regularizer.name, regularizer(seed_inputs))
                                   for regularizer in regularizers]
                regularized_score_values = [
                    (-1. * score_value) + sum([v for _, v in regularizations])
                    for score_value in score_values
                ]
                if mixed_precision_model:
                    regularized_score_values = [
                        optimizer.get_scaled_loss(score_value)
                        for score_value in regularized_score_values
                    ]
            grads = tape.gradient(regularized_score_values,
                                  seed_inputs,
                                  unconnected_gradients=unconnected_gradients)
            grads = listify(grads)
            if mixed_precision_model:
                grads = optimizer.get_unscaled_gradients(grads)
            if gradient_modifier is not None:
                grads = (gradient_modifier(g) for g in grads)
            optimizer.apply_gradients(zip(grads, seed_inputs))

            for callback in callbacks:
                callback(i,
                         self._apply_clip(seed_inputs, input_ranges),
                         grads,
                         score_values,
                         outputs,
                         regularizations=regularizations,
                         overall_score=regularized_score_values)

        for callback in callbacks:
            callback.on_end()

        cliped_value = self._apply_clip(seed_inputs, input_ranges)
        if len(self.model.inputs) == 1 and (seed_input is None or not isinstance(seed_input, list)):
            cliped_value = cliped_value[0]

        return cliped_value

    def _get_optimizer(self, optimizer, mixed_precision_model):
        if optimizer is None:
            optimizer = tf.optimizers.RMSprop(1.0, 0.95)
        if mixed_precision_model:
            try:
                # Wrap optimizer
                optimizer = LossScaleOptimizer(optimizer)
            except ValueError as e:
                raise ValueError(
                    ("The same `optimizer` instance should be NOT used twice or more."
                     " You can be able to avoid this error by creating new optimizer instance"
                     " each calling __call__().")) from e
        return optimizer

    def _get_input_ranges(self, input_range):
        input_ranges = listify(input_range,
                               return_empty_list_if_none=False,
                               convert_tuple_to_list=False)
        if len(input_ranges) == 1 and len(self.model.inputs) > 1:
            input_ranges = input_ranges * len(self.model.inputs)
        input_ranges = [(None, None) if r is None else r for r in input_ranges]
        for i, r in enumerate(input_ranges):
            if len(r) != 2:
                raise ValueError(
                    'The length of input range tuple must be 2 (Or it is just `None`, not tuple), '
                    'but you passed {} as `input_ranges[{}]`.'.format(r, i))
        return input_ranges

    def _get_seed_inputs(self, seed_inputs, input_ranges):
        # Prepare seed_inputs
        if seed_inputs is None or len(seed_inputs) == 0:
            # Replace None to 0.0-1.0 or any properly value
            input_ranges = ((0., 1.) if low is None and high is None else (low, high)
                            for low, high in input_ranges)
            input_ranges = ((high - np.abs(high / 2.0), high) if low is None else (low, high)
                            for low, high in input_ranges)
            input_ranges = ((low, low + np.abs(low * 2.0)) if high is None else (low, high)
                            for low, high in input_ranges)
            input_ranges = list(input_ranges)
            # Prepare input_shape
            input_shapes = (input_tensor.shape[1:] for input_tensor in self.model.inputs)
            # Generae seed-inputs
            seed_inputs = (tf.random.uniform(shape, low, high)
                           for (low, high), shape in zip(input_ranges, input_shapes))
        else:
            seed_inputs = listify(seed_inputs)
        # Convert numpy to tf-tensor
        seed_inputs = (tf.constant(X, dtype=input_tensor.dtype)
                       for X, input_tensor in zip(seed_inputs, self.model.inputs))
        # Do expand_dims when tensor doesn't have the dim for samples
        seed_inputs = (tf.expand_dims(X, axis=0) if len(X.shape) < len(input_tensor.shape) else X
                       for X, input_tensor in zip(seed_inputs, self.model.inputs))
        seed_inputs = list(seed_inputs)
        if len(seed_inputs) != len(self.model.inputs):
            raise ValueError(
                ("The lengths of seed_inputs and model's inputs don't match."
                 " seed_inputs: {}, model's inputs: {}").format(len(seed_inputs),
                                                                len(self.model.inputs)))
        return seed_inputs

    def _get_input_modifiers(self, input_modifier):
        input_modifiers = self._get_dict(input_modifier, keys=self.model.input_names)
        if len(input_modifiers) != len(self.model.inputs):
            raise ValueError('The model has {} inputs, but you passed {} as input_modifiers. '
                             'When the model has multiple inputs, '
                             'you must pass a dictionary as input_modifiers.'.format(
                                 len(self.model.inputs), input_modifier))
        return input_modifiers

    def _get_regularizers(self, regularizer):
        regularizers = listify(regularizer)
        return regularizers

    def _get_dict(self, values, keys):
        if isinstance(values, dict):
            _values = defaultdict(list, values)
            for key in keys:
                _values[key] = listify(_values[key])
        else:
            _values = defaultdict(list)
            values = listify(values)
            for k in keys:
                _values[k] = values
        return _values

    def _apply_clip(self, seed_inputs, input_ranges):
        input_ranges = [(input_tensor.dtype.min if low is None else low,
                         input_tensor.dtype.max if high is None else high)
                        for (low, high), input_tensor in zip(input_ranges, self.model.inputs)]
        clipped_values = (np.array(K.clip(X, low, high))
                          for X, (low, high) in zip(seed_inputs, input_ranges))
        clipped_values = [
            X.astype(np.int) if isinstance(t, int) else X.astype(np.float)
            for X, (t, _) in zip(clipped_values, input_ranges)
        ]
        return clipped_values
