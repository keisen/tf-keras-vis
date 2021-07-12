import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from deprecated import deprecated
from packaging.version import parse as version

MAX_STEPS = 'TF_KERAS_VIS_MAX_STEPS'


@deprecated(version='0.7.0')
def check_steps(steps):
    return get_num_of_steps_allowed(steps)


def get_num_of_steps_allowed(steps) -> int:
    return min(steps, int(os.environ[MAX_STEPS])) if MAX_STEPS in os.environ else steps


def num_of_gpus() -> Tuple[int, int]:
    """Return the number of physical and logical gpus.

    Returns:
        Tuple[int, int]: A tuple of the number of physical and logical gpus.
    """
    if version(tf.version.VERSION) < version("2.1.0"):
        list_physical_devices = tf.config.experimental.list_physical_devices
        list_logical_devices = tf.config.experimental.list_logical_devices
    else:
        list_physical_devices = tf.config.list_physical_devices
        list_logical_devices = tf.config.list_logical_devices
    physical_gpus = list_physical_devices('GPU')
    if physical_gpus:
        logical_gpus = list_logical_devices('GPU')
        return len(physical_gpus), len(logical_gpus)
    else:
        return 0, 0


def listify(value, return_empty_list_if_none=True, convert_tuple_to_list=True) -> list:
    """Ensures that the value is a list.

    If it is not a list, it creates a new list with `value` as an item.

    Args:
        value (object): A list or something else.
        return_empty_list_if_none (bool, optional): When True (default), None you passed as `value`
            will be converted to a empty list (i.e., `[]`). When False, None will be converted to
            a list that has an None (i.e., `[None]`). Defaults to True.
        convert_tuple_to_list (bool, optional): When True (default), a tuple you passed as `value`
            will be converted to a list. When False, a tuple will be unconverted
            (i.e., returning a tuple object that was passed as `value`). Defaults to True.
    Returns:
        list: A list. When `value` is a tuple and `convert_tuple_to_list` is False, a tuple.
    """
    if not isinstance(value, list):
        if value is None and return_empty_list_if_none:
            value = []
        elif isinstance(value, tuple) and convert_tuple_to_list:
            value = list(value)
        else:
            value = [value]
    return value


def standardize(array, value_range=(1., 0.)) -> np.ndarray:
    """Standardization.

    Args:
        array (np.ndarray): A tensor.
        value_range (tuple, optional): `array` will be scaled in this range. Defaults to (1., 0.).

    Returns:
        np.ndarray: Standardized array.
    """
    max_value = np.max(array, axis=tuple(range(array.ndim)[1:]), keepdims=True)
    min_value = np.min(array, axis=tuple(range(array.ndim)[1:]), keepdims=True)
    normalized_array = (array - min_value) / (max_value - min_value + K.epsilon())
    return normalized_array


@deprecated(version='0.6.0', reason="Inappropriate naming, Use `standardize()` instead.")
def normalize(array, value_range=(1., 0.)):
    if value_range is None:
        return standardize(array)
    else:
        normalized_array = standardize(array)
        high, low = value_range
        return (high - low) * normalized_array + low


def find_layer(model, condition, offset=None, reverse=True) -> tf.keras.layers.Layer:
    found_offset = offset is None
    for layer in reversed(model.layers):
        if not found_offset:
            found_offset = (layer == offset)
        if condition(layer) and found_offset:
            return layer
        if isinstance(layer, tf.keras.Model):
            if found_offset:
                result = find_layer(layer, condition, offset=None, reverse=reverse)
            else:
                result = find_layer(layer, condition, offset=offset, reverse=reverse)
            if result is not None:
                return result
    return None


def zoom_factor(from_shape, to_shape) -> Tuple[float, float]:
    return tuple(float(t) / float(f) for f, t in zip(from_shape, to_shape))


def is_mixed_precision(model) -> bool:
    """Check whether the model has any lower precision variable or not.

    Args:
        model (tf.keras.Model): A model instance.

    Returns:
        bool: When the model has any lower precision variable, True.
    """
    if version(tf.version.VERSION) >= version("2.4.0"):
        for layer in model.layers:
            if (layer.compute_dtype == tf.float16) or \
               (isinstance(layer, tf.keras.Model) and is_mixed_precision(layer)):
                return True
    return False


@deprecated(version='0.7.0', reason="Unnecessary function.")
def lower_precision_dtype(model):
    if version(tf.version.VERSION) >= version("2.4.0"):
        for layer in model.layers:
            if (layer.compute_dtype in [tf.float16, tf.bfloat16]) or \
               (isinstance(layer, tf.keras.Model) and is_mixed_precision(layer)):
                return layer.compute_dtype
    return model.dtype  # pragma: no cover
