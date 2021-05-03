import os
import shutil

import pytest
import tensorflow as tf
from packaging.version import parse as version
from tensorflow.keras.layers import (Conv2D, Dense, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.models import Model, load_model

from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.scores import CategoricalScore

if version(tf.version.VERSION) >= version("2.4.0"):
    from tensorflow.keras.mixed_precision import set_global_policy


def model(policy_name=None):
    if policy_name is not None:
        set_global_policy(policy_name)
    inputs = Input((8, 8, 3))
    x = Conv2D(2, 3, activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x)


@pytest.fixture(scope="function", autouse=True)
def model_with_default_policy():
    return model(policy_name='float32')


@pytest.fixture(scope="function", autouse=True)
def model_with_mixed_precision_policy():
    return model(policy_name='mixed_float16')


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
def test__call__(model_with_default_policy):
    activation_maximization = ActivationMaximization(model_with_default_policy)
    result = activation_maximization([CategoricalScore(1)], steps=1)
    assert result.shape == (1, 8, 8, 3)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
def test__call__with_mixed_precison(model_with_mixed_precision_policy):
    activation_maximization = ActivationMaximization(model_with_mixed_precision_policy)
    result = activation_maximization([CategoricalScore(1)], steps=1)
    assert result.shape == (1, 8, 8, 3)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
def test__call__with_saved_model(model_with_mixed_precision_policy, tmpdir):
    # Save model
    model_path = os.path.join(tmpdir, 'model_with_mixed_precision_policy.h5')
    model_with_mixed_precision_policy.save(model_path)
    # Load model
    set_global_policy('float32')
    model_with_default_policy = load_model(model_path)
    # Test
    activation_maximization = ActivationMaximization(model_with_default_policy)
    result = activation_maximization([CategoricalScore(1)], steps=1)
    assert result.shape == (1, 8, 8, 3)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
def test__call__twice(model_with_mixed_precision_policy):
    # Test No.1
    activation_maximization = ActivationMaximization(model_with_mixed_precision_policy)
    result = activation_maximization([CategoricalScore(1)], steps=1)
    assert result.shape == (1, 8, 8, 3)
    # Test No.2
    activation_maximization = ActivationMaximization(model_with_mixed_precision_policy)
    result = activation_maximization([CategoricalScore(1)], steps=1)
    assert result.shape == (1, 8, 8, 3)


@pytest.mark.skipif(version(tf.version.VERSION) < version("2.4.0"),
                    reason="This test is enabled when tensorflow version is 2.4.0+.")
def test__call__twice_with_same_optimizer(model_with_mixed_precision_policy):
    optimizer = tf.optimizers.RMSprop(1.0, 0.95)
    # Test No.1
    activation_maximization = ActivationMaximization(model_with_mixed_precision_policy)
    result = activation_maximization([CategoricalScore(1)], steps=1, optimizer=optimizer)
    assert result.shape == (1, 8, 8, 3)
    # Test No.2
    activation_maximization = ActivationMaximization(model_with_mixed_precision_policy)
    try:
        result = activation_maximization([CategoricalScore(1)], steps=1, optimizer=optimizer)
        raise AssertionError('Value Error was NOT occurred.')
    except ValueError:
        pass
