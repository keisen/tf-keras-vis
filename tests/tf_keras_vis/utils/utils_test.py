import os

import pytest
import tensorflow as tf
from packaging.version import parse as version

from tf_keras_vis.utils import MAX_STEPS, find_layer, get_num_of_steps_allowed, num_of_gpus


class TestUtils():
    @pytest.mark.parametrize("env,steps,expected", [
        (None, -1, -1),
        (None, 0, 0),
        (None, 1, 1),
        (None, 2, 2),
        (None, 100, 100),
        (1, -1, -1),
        (1, 0, 0),
        (1, 1, 1),
        (1, 2, 1),
        (1, 100, 1),
        (2, -1, -1),
        (2, 0, 0),
        (2, 1, 1),
        (2, 2, 2),
        (2, 100, 2),
    ])
    def test_get_num_of_steps_allowed(self, env, steps, expected):
        _env = os.environ.get(MAX_STEPS)
        try:
            if env is None:
                os.environ.pop(MAX_STEPS, None)
            else:
                os.environ[MAX_STEPS] = str(env)
            actual = get_num_of_steps_allowed(steps)
            assert actual == expected
        finally:
            if _env is not None:
                os.environ[MAX_STEPS] = _env

    def test_num_of_gpus_if_no_gpus(self, monkeypatch):
        def list_physical_devices(name):
            return None

        def list_logical_devices(name):
            return None

        if version(tf.version.VERSION) < version("2.1.0"):
            monkeypatch.setattr(tf.config.experimental, "list_physical_devices",
                                list_physical_devices)
            monkeypatch.setattr(tf.config.experimental, "list_logical_devices",
                                list_logical_devices)

        else:
            monkeypatch.setattr(tf.config, "list_physical_devices", list_physical_devices)
            monkeypatch.setattr(tf.config, "list_logical_devices", list_logical_devices)
        a, b, = num_of_gpus()
        assert a == 0
        assert b == 0

    def test_num_of_gpus(self, monkeypatch):
        def list_physical_devices(name):
            return ['dummy-a', 'dummy-b']

        def list_logical_devices(name):
            return ['a1', 'a2', 'b1', 'b2']

        if version(tf.version.VERSION) < version("2.1.0"):
            monkeypatch.setattr(tf.config.experimental, "list_physical_devices",
                                list_physical_devices)
            monkeypatch.setattr(tf.config.experimental, "list_logical_devices",
                                list_logical_devices)

        else:
            monkeypatch.setattr(tf.config, "list_physical_devices", list_physical_devices)
            monkeypatch.setattr(tf.config, "list_logical_devices", list_logical_devices)
        a, b, = num_of_gpus()
        assert a == 2
        assert b == 4

    @pytest.mark.parametrize("offset_of_child_layer", [
        False,
        True,
    ])
    def test_find_layer(self, offset_of_child_layer, conv_model):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, 3, padding='same', input_shape=(8, 8, 3)),
            conv_model,
            tf.keras.layers.Dense(1),
        ])
        offset = conv_model.layers[-1] if offset_of_child_layer else None
        actual = find_layer(model, lambda l: l.name == 'conv_1', offset=offset)
        assert conv_model.get_layer(name='conv_1') == actual
