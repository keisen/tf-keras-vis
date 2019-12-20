# tf-keras-vis
tf-keras-vis is a visualization toolkit for debugging Keras models with Tensorflow 2.0, but not original Keras.

The features of tf-keras-vis are based on [keras-vis](https://github.com/raghakot/keras-vis), but tf-keras-vis's APIs doesn't have compatibility with keras-vis's because, instead of getting it, we prioritized to get following features.

- Support processing multipul images at a time as a batch
- Support tf.keras.Model that has multipul inputs (and, of course, multipul outpus too)
- Allow use optimizers that embeded in tf.keras

And then we will add some algorisms such as below.

- [SmoothGrad: removing noise by adding noise](https://arxiv.org/pdf/1706.03825.pdf) (DONE)
- Deep Dream
- Style transfer


## Requirements

* Python 3.6+
* (tensorflow or tensorflow-gpu) >= 2.0


## Installation

* PyPI

```bash
$ pip install tf-keras-vis
```

* Sources

```bash
$ cd tf-keras-vis
$ pip install -e .
```

Or

```bash
$ cd tf-keras-vis
$ python setup.py install
```


## Usage

T.B.D.

For now, Please see `examples/activation_maximization.ipynb` and `examples/saliency.ipynb`.
When you want to run jupyter notebook, we recommend that install tf-keras-vis such as follow:

```bash
$ cd tf-keras-vis
$ pip install -e .[examples]
```


## API Documentation

T.B.D


## Known Issues

* With InceptionV3 ActivationMaximization doesn't work well, that's, it might generate meanninglessly bulr image.
* With cascading model gradcam doesn't work well, that's, it might occur some error.
