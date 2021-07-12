import warnings
from collections import defaultdict
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from .. import ModelVisualization
from ..utils import get_num_of_steps_allowed, is_mixed_precision, listify
from ..utils.regularizers import LegacyRegularizer
from .callbacks import managed_callbacks
from .input_modifiers import Jitter, Rotate2D
from .regularizers import Norm, TotalVariation2D


class ActivationMaximization(ModelVisualization):
    """ActivationMaximization.
    """
    def __call__(
            self,
            score,
            seed_input=None,
            input_range=(0, 255),
            input_modifiers=[Jitter(jitter=4), Rotate2D(degree=1)],
            regularizers=[TotalVariation2D(weight=1.0),
                          Norm(weight=0.3, p=1)],
            steps=200,
            optimizer=None,  # When None, the default is tf.optimizers.RMSprop(1.0, 0.999)
            normalize_gradient=None,  # Disabled option.
            gradient_modifier=None,
            callbacks=None,
            training=False,
            unconnected_gradients=tf.UnconnectedGradients.NONE,
            activation_modifiers=None,
            _compatible_mode=False,  # Hidden option.
    ) -> Union[np.ndarray, list]:
        """Generate the model inputs that maximize the output of the given `score` functions.

        By default, this method is optimized to visualize `tf.keras.application.VGG16` model.
        So if you want to visualize other models, you have to tune the parameters of this method.

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
                When `None`, the seed_input value will be automatically generated from a uniform
                distribution. If you want to visualize multiple images (i.e., batch_size > 1),
                you have to pass a seed_input object. For example::

                    seed_input = tf.random.uniform((samples, ..., channels), low, high)

                Farthermore, if the model has multiple inputs and you want multiple images,
                you have to do as follows::

                    seed_input = [
                        tf.random.uniform((samples, ..., channels), low, high),  # 1st input
                        tf.random.uniform((samples, ..., channels), low, high),  # 2nd input
                        ...
                    ]

                Defaults to None.
            input_range: A tuple of two int values or a list of them. The tuple indicates
                `(min, max)` values that is range of the result of this method. If the model has
                multiple inputs, you can use different input ranges for each model input by
                passing list of tuples. For example::

                    input_range = [
                        (0, 255),     # The 1st model input's range
                        (-1.0, 1.0),  # The 2nd model input's range
                        ...
                    ]

                When `None` or `(None, None)` tuple, the input tensor
                (i.e., the result of this method) will be not applied any limitation.
                Defaults to (0, 255).
            input_modifiers: A :obj:`tf_keras_vis.activation_maximization.input_modifiers.
                InputModifier` instance, a function, a list of them when the model has a single
                input. For example::

                    input_modifiers = [Jitter(jitter=8), Rotate(degree=3), Scale(high=1.1)]

                When the model has multiple inputs, you have to pass a dictionary
                that contains the lists of input modifiers for each model inputs::

                    input_modifiers = {
                        'input_1': [Jitter(jitter=8), Rotate(degree=3), Scale(high=1.1)],
                        'input_2': [Jitter(jitter=8)],
                        ...
                    }

                Or you could also pass a list of lists of input modifiers for each model inputs as
                follows::

                    input_modifiers = [
                        [Jitter(jitter=8), Rotate(degree=3), Scale(high=1.1)],  # For 1st input
                        [Jitter(jitter=8)],                                     # For 2nd input
                        ...
                    ]

                Defaults to [Jitter(jitter=4), Rotate(degree=1)].
            regularizers: A :obj:`tf_keras_vis.utils.regularizers.Regularizer` instance,
                a list of regularizers or a list that has lists of regularizers for each input.
                For example::

                    regularizers = [TotalVariation2D(weight=1.0), Norm(weight=0.3, p=1)]

                > Please notice that `regularizes` does NOT accept function object like
                `input_modifiers`.

                When the model has multiple inputs, you have to pass a dictionary
                that contains the lists of regularizers for each model inputs::

                    regularizers = {
                        'input_1': [TotalVariation2D(weight=1.0), Norm(weight=0.3, p=1)],
                        'input_2': [Norm(weight=1.0, p=2)],
                        ...
                    }

                Or you could also pass a list of lists of regularizers for each model inputs as
                follows::

                    regularizers = [
                        [TotalVariation2D(weight=1.0), Norm(weight=0.3, p=1)],  # For 1st input
                        [Norm(weight=1.0, p=2)],                               # For 2nt input
                        ...
                    ]

                Defaults to [TotalVariation2D(weight=1.0), Norm(weight=0.3, p=1)].
            steps: The number of gradient descent iterations. Defaults to 200.
            optimizer: A `tf.optimizers.Optimizer` instance. When None, it will be automatically
                created. Defaults to `tf.optimizers.RMSprop(learning_rate=1.0, rho=0.999)`.
            gradient_modifier: A function to modify gradients.
                Defaults to None.
            callbacks: A :obj:`tf_keras_vis.activation_maximization.callbacks.Callback` instance
                or a list of them.
                Defaults to None.
            training: A bool that indicates whether the model's training-mode on or off.
                Defaults to False.
            unconnected_gradients: Specifies the gradient value returned when the given input
                tensors are unconnected.
                Defaults to tf.UnconnectedGradients.NONE.
            activation_modifiers: A function or a dictionary of them (the key is input layer's
                name). When the model has multiple inputs, you have to pass a dictionary::

                    activation_modifiers = {
                        'input_1': lambda x: ...,
                        'input_2': lambda x: ...,
                        ...
                    }

                This functions will be executed before returning the result. Defaults to None.
        Returns:
            An :obj:`numpy.ndarray` when the model has a single input.
            When the model has multiple inputs, a list of :obj:`numpy.ndarray`.

        Raises:
            :obj:`ValueError`: When there is any invalid arguments.
        """
        arguments = dict(
            (k, v) for k, v in locals().items() if k != 'self' and not k.startswith('_'))

        if normalize_gradient is not None:
            warnings.warn(
                "`normalize_gradient` option of ActivationMaximization#__call__() is disabled.,"
                " And this will be removed in future.", DeprecationWarning)

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

        # activation_modifiers
        activation_modifiers = self._get_activation_modifiers(activation_modifiers)

        with managed_callbacks(**arguments) as callbacks:
            input_values = seed_inputs
            input_variables = [tf.Variable(X) for X in input_values]
            for step in range(get_num_of_steps_allowed(steps)):
                # Modify input values
                for i, name in enumerate(self.model.input_names):
                    for modifier in input_modifiers[name]:
                        input_values[i] = modifier(input_values[i])

                # Copy input values to variables
                if _compatible_mode:
                    input_variables = [
                        tf.Variable(tf.cast(X, tf.float16) if mixed_precision_model else X)
                        for X in input_values
                    ]
                else:
                    for V, X in zip(input_variables, input_values):
                        V.assign(X)

                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(input_variables)
                    if _compatible_mode:
                        input_values = input_variables
                    else:
                        input_values = [V.value() for V in input_variables]
                    # Calculate scores
                    outputs = self.model(input_values, training=training)
                    outputs = listify(outputs)
                    score_values = self._calculate_scores(outputs, scores)
                    # Calculate regularization
                    regularization_values, regularized_score_values = \
                        self._calculate_regularization(regularizers, input_values, score_values)
                    # Scale loss
                    if mixed_precision_model:
                        regularized_score_values = [
                            optimizer.get_scaled_loss(score_value)
                            for score_value in regularized_score_values
                        ]
                # Calculate gradients and Update variables
                grads = tape.gradient(regularized_score_values,
                                      input_variables,
                                      unconnected_gradients=unconnected_gradients)
                grads = listify(grads)
                if mixed_precision_model:
                    grads = optimizer.get_unscaled_gradients(grads)
                if gradient_modifier is not None:
                    grads = [gradient_modifier(g) for g in grads]
                optimizer.apply_gradients(zip(grads, input_variables))

                # Update input values
                input_values = [V.value() for V in input_variables]
                if _compatible_mode and mixed_precision_model:
                    input_values = [tf.cast(X, tf.float32) for X in input_values]

                # Calculate clipped values
                clipped_value = self._clip_and_modify(input_values, input_ranges,
                                                      activation_modifiers)

                # Execute callbacks
                for callback in callbacks:
                    callback(step,
                             clipped_value,
                             grads,
                             score_values,
                             outputs,
                             regularizations=regularization_values,
                             overall_score=regularized_score_values)

        if len(self.model.inputs) == 1 and (seed_input is None
                                            or not isinstance(seed_input, list)):
            clipped_value = clipped_value[0]
        return clipped_value

    def _calculate_regularization(self, regularizers, seed_inputs, score_values):
        if isinstance(regularizers, list):
            regularization_values = [(regularizer.name, regularizer(seed_inputs))
                                     for regularizer in regularizers]
        else:
            regularization_values = ([
                (name, regularizer(seed_inputs[i]))
                for name, regularizer in regularizers[input_layer_name].items()
            ] for i, input_layer_name in enumerate(self.model.input_names))
            regularization_values = sum(regularization_values, [])
        regularized_score_values = [-1.0 * score_value for score_value in score_values]
        regularized_score_values += [value for _, value in regularization_values]
        return regularization_values, regularized_score_values

    def _get_optimizer(self, optimizer, mixed_precision_model):
        if optimizer is None:
            optimizer = tf.optimizers.RMSprop(learning_rate=1.0, rho=0.999)
        if mixed_precision_model:
            try:
                # Wrap optimizer
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            except ValueError as e:
                raise ValueError(
                    "The same `optimizer` instance should be NOT used twice or more."
                    " You can be able to avoid this error by creating new optimizer instance"
                    " each calling __call__().") from e
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
                    "The length of input range tuple must be 2 (Or it is just `None`, not tuple), "
                    f"but you passed {r} as `input_ranges[{i}]`.")
            a, b = r
            if None not in r and type(a) != type(b):
                raise TypeError(
                    "The type of low and high values in the input range must be the same, "
                    f"but you passed {r} are {type(a)} and {type(b)} ")
        return input_ranges

    def _get_seed_inputs(self, seed_inputs, input_ranges):
        # Prepare seed_inputs
        seed_inputs = listify(seed_inputs)
        if len(seed_inputs) == 0:
            # Replace None to 0.0-1.0 or any properly value
            input_ranges = ((0., 1.) if low is None and high is None else (low, high)
                            for low, high in input_ranges)
            input_ranges = ((high - np.abs(high / 2.0), high) if low is None else (low, high)
                            for low, high in input_ranges)
            input_ranges = ((low, low + np.abs(low * 2.0)) if high is None else (low, high)
                            for low, high in input_ranges)
            # Prepare input_shape
            input_shapes = (input_tensor.shape[1:] for input_tensor in self.model.inputs)
            # Generae seed-inputs
            seed_inputs = (tf.random.uniform(shape, low, high)
                           for (low, high), shape in zip(input_ranges, input_shapes))
        # Convert numpy to tf-tensor
        seed_inputs = (tf.cast(tf.constant(X), dtype=input_tensor.dtype)
                       for X, input_tensor in zip(seed_inputs, self.model.inputs))
        # Do expand_dims when an seed_input doesn't have the dim for samples
        seed_inputs = (tf.expand_dims(X, axis=0) if len(X.shape) < len(input_tensor.shape) else X
                       for X, input_tensor in zip(seed_inputs, self.model.inputs))
        seed_inputs = list(seed_inputs)
        if len(seed_inputs) != len(self.model.inputs):
            raise ValueError(
                "The lengths of seed_inputs and model's inputs don't match."
                f" seed_inputs: {len(seed_inputs)}, model's inputs: {len(self.model.inputs)}")
        return seed_inputs

    def _get_input_modifiers(self, input_modifier):
        return self._get_callables_to_apply_to_each_input(input_modifier, "input modifiers")

    def _get_regularizers(self, regularizer):
        legacy_regularizers = self._get_legacy_regularizers(regularizer)
        if legacy_regularizers is not None:
            warnings.warn(
                "`tf_keras_vis.utils.regularizers.Regularizer` is deprecated. "
                "Use tf_keras_vis.activation_maximization.regularizers.Regularizer instead.",
                DeprecationWarning)
            return legacy_regularizers
        else:
            regularizers = self._get_callables_to_apply_to_each_input(regularizer, "regularizers")
            regularizers = ((input_layer_name,
                             self._define_regularizer_names(regularizer_list, input_layer_name))
                            for input_layer_name, regularizer_list in regularizers.items())
            return defaultdict(dict, regularizers)

    def _define_regularizer_names(self, regularizers, input_layer_name):
        regularizers = ((f"regularizer-{i}", regularizer)
                        for i, regularizer in enumerate(regularizers))
        regularizers = (((regularizer.name if hasattr(regularizer, 'name') else name), regularizer)
                        for name, regularizer in regularizers)
        if len(self.model.input_names) > 1:
            regularizers = ((f"{name}({input_layer_name})", regularizer)
                            for name, regularizer in regularizers)
        return defaultdict(list, regularizers)

    def _get_legacy_regularizers(self, regularizer):
        if isinstance(regularizer, dict):
            _regularizer = [listify(r) for r in regularizer.values()]
        else:
            _regularizer = regularizer
        if isinstance(_regularizer, (tuple, list)):
            if any(isinstance(r, (tuple, list)) for r in _regularizer):
                has_legacy = ((isinstance(r, LegacyRegularizer) for r in listify(_regularizers))
                              for _regularizers in _regularizer)
                has_legacy = (any(_legacy) for _legacy in has_legacy)
                if any(has_legacy):
                    raise ValueError(
                        "Legacy Regularizer instances (that inherits "
                        "`tf_keras_vis.utils.regularizers.LegacyRegularizer`) must be "
                        "passed to ActivationMaximization#__call__() "
                        "in the form of a instance or a list of instances. "
                        "Please modify the `regularizer` argument or "
                        "change the inheritance source to "
                        "`tf_keras_vis.activation_maximization.regularizers.Regularizer`"
                        f" regularizer: {regularizer}")
            else:
                has_legacy = [isinstance(r, LegacyRegularizer) for r in _regularizer]
                if all(has_legacy):
                    return _regularizer
                if any(has_legacy):
                    raise ValueError(
                        "the regularizer instance (that inherits "
                        "`tf_keras_vis.activation_maximization.regularizers.Regularizer`) "
                        "and legacy regularizer (that inherits "
                        "`tf_keras_vis.utils.regularizers.LegacyRegularizer` can NOT be mixed."
                        f" regularizer: {regularizer}")
        elif isinstance(_regularizer, LegacyRegularizer):
            return listify(_regularizer)
        return None

    def _get_callables_to_apply_to_each_input(self, callables, object_name):
        keys = self.model.input_names
        if isinstance(callables, dict):
            non_existent_keys = set(callables.keys()) - set(keys)
            if len(non_existent_keys) > 0:
                raise ValueError(
                    f"The model inputs are `{keys}`. However the {object_name} you passed have "
                    f"non existent input name: `{non_existent_keys}`")
            callables = ((k, listify(v)) for k, v in callables.items())
        else:
            callables = listify(callables)
            if len(callables) == 0 or len(list(filter(lambda x: type(x) == list, callables))) == 0:
                callables = [callables]
            if len(callables) <= len(keys):
                callables = (listify(value_each_input) for value_each_input in callables)
                callables = zip(keys, callables)
            else:
                raise ValueError(f"The number of model's inputs are {len(keys)},"
                                 f" but you define {len(callables)} {object_name}.")
        return defaultdict(list, callables)

    def _get_activation_modifiers(self, activation_modifiers):
        if isinstance(activation_modifiers, dict):
            non_existent_names = set(activation_modifiers.keys()) - set(self.model.input_names)
            if len(non_existent_names) > 0:
                raise ValueError(f"The model inputs are `{self.model.input_names}`. "
                                 "However the activation modifiers you passed have "
                                 f"non existent input names: `{non_existent_names}`")
        else:
            activation_modifiers = {self.model.input_names[0]: activation_modifiers}
        return defaultdict(lambda: None, activation_modifiers)

    def _clip_and_modify(self, seed_inputs, input_ranges, activation_modifiers):
        input_ranges = [(input_tensor.dtype.min if low is None else low,
                         input_tensor.dtype.max if high is None else high)
                        for (low, high), input_tensor in zip(input_ranges, self.model.inputs)]
        clipped_values = (np.array(K.clip(X, low, high))
                          for X, (low, high) in zip(seed_inputs, input_ranges))
        clipped_values = (X.astype(np.int) if isinstance(t, int) else X.astype(np.float)
                          for X, (t, _) in zip(clipped_values, input_ranges))
        if activation_modifiers is not None:
            clipped_values = ((activation_modifiers[name], seed_input)
                              for name, seed_input in zip(self.model.input_names, clipped_values))
            clipped_values = (seed_input if modifier is None else modifier(seed_input)
                              for modifier, seed_input in clipped_values)
        return list(clipped_values)
