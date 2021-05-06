import tensorflow as tf
from packaging.version import parse as version

from tf_keras_vis.utils import num_of_gpus


class TestUtils():
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