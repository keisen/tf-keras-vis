# tf-keras-vis
tf-keras-vis is a visualization toolkit for debugging Keras models.

This toolkit's concepts is based on [keras-vis](https://github.com/raghakot/keras-vis) that is a grate useful toolkit for Deep learning scientists and engineers. tf-keras has been developed to add some concepts such as below for achieving our works.

- As the same suggests supporting `tf.kearas`, but not original Keras.
- Support Keras model that has multipul inputs and multipul outpus
- Support processing multipul images as a batch

And then we will add some algorisms such as below.

- [SmoothGrad: removing noise by adding noise](https://arxiv.org/pdf/1706.03825.pdf) (DONE)
- Deep Dream
- Style transfer


## Requirements

* Python 3.6+
* tensorflow >= 2.0 Or tensorflow-gpu >= 2.0


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

or

```bash
$ cd tf-keras-vis
$ python setup.py install
```


## Usage

For now, Please see `examples/sandbox.ipynb`.

T.B.D


## API Documentation

T.B.D
