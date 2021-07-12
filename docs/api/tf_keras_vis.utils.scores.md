Scores
==================================

The Scores are used to specify somethings you want to visualized with Saliency, X-CAMs and ActivationMaximization.
Here, we will introduce the fundamental score usage with the models below.

* Categorical classification model
* Binary classification model
* Regression model

If you want just to see the API specification, please skip to
[Classes section](tf_keras_vis.utils.scores.html#module-tf_keras_vis.utils.scores).


## Categorical classification model

We expect the output shape of the categorical classification model is `(batch_size, num_of_classes)`. That's, `softmax` activation function is applied to the output layer of the model.

The `output` variable below is assigned model output value. The code snippet below means that it will return the values corresponding to the 20th of categories.

```python
def score_function(output): # output shape is (batch_size, num_of_classes)
   return output[:, 20]
```

The function below means the same as above. When you don't need to implement additional code to process output value, we recommend you to utilize the [CategoricalScore](tf_keras_vis.utils.scores.html#module-tf_keras_vis.utils.scores.CategoricalScore).

```python
from tf_keras_vis.utils.scores import CategoricalScore

score = CategoricalScore(20)
```

If you want to visualize corresponding to multiple various categories, you can define as follows. The code snippet below means that it will return the three values for the 20th, the 48th and the No.123rd of categories respectively.

```python
def score_function(output): # output shape is (batch_size, num_of_classes)
   return (output[0, 20], output[1, 48], output[0, 128])
```

```{note}
Please note that the length of the values returned by the score function MUST be identical to `batch_size` (the number of samples).
```

Of course, you can also use [CategoricalScore](tf_keras_vis.utils.scores.html#module-tf_keras_vis.utils.scores.CategoricalScore).

```python
from tf_keras_vis.utils.scores import CategoricalScore

score = CategoricalScore([20, 48, 128])
```


## Binary classification task

We expect the output shape of the binary classification model is `(batch_size, 1)` and the output value range is `[0, 1]`. That's, `sigmoid` activation function is applied to the output layer of the model.

In categorical classification, the score functions just return the values corresponding to somethings you want to visualize. However, in binary classification, you need to be aware of whether the value you want to visualize is 0.0 or 1.0 (False or True).

### 1.0 (True)

Like the categorical classification, it just returns the value as follows.

```python
def score_function(output): # output shape is (batch_size, 1)
   return output[:, 0]
```

### 0.0 (False)

The model output value smaller, the score value should be larger, so you need to multiply by `-1.0`.

```python
def score_function(output): # output shape is (batch_size, 1)
   return -1.0 * output[:, 0]
```


### Utilizing BinaryScore class

Of course, we recommend you to utilize BinaryScore class as follows.

```python
from tf_keras_vis.utils.scores import BinaryScore
score = BinaryScore(0.0) # or BinaryScore(False) 
```

or

```python
from tf_keras_vis.utils.scores import BinaryScore
score = BinaryScore(1.0) # or BinaryScore(True) 
```


## Regression task

We expect the output shape of the regression model is `(batch_size, 1)` like binary classification, however the output value range is no limitation. That's, `linear` activation function is applied to the output layer of the model.

In regression task, we need to consider how what we want to visualize contributes to the model output.
Here, we introduce a simple way each for three situations below we want to visualize.

1. Increase the output value
2. Decrease the output value
3. Maintain the output value at ...


### 1. Increase the output value

It just returns the value like the categorical classification.

```python
def score_function(output):
    return output[:, 0]
```

### 2. Decrease the output value

The model output value smaller, the score value should be larger,
so you need to multiply by `-1.0`.

```python
def score_function(output):
    return -1.0 * output[:, 0]
```

### 3. Maintain the output value at ...

The model output value closer to the target value, the score value should be larger, so you need to calculate `abs(1.0 / (target_value - model_output_value))`.
For example, suppose the target value is 0.0, the score function should be as follows.

```python
def score_function(output):
    return tf.math.abs(1.0 / (output[:, 0] + tf.keras.backend.epsilon()))
```


Classes
-----------------

```{eval-rst}
.. automodule:: tf_keras_vis.utils.scores
   :members:
   :show-inheritance:
```
