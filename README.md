# tf-keras-vis
[![Downloads](https://pepy.tech/badge/tf-keras-vis)](https://pepy.tech/project/tf-keras-vis)
[![PyPI version](https://badge.fury.io/py/tf-keras-vis.svg)](https://badge.fury.io/py/tf-keras-vis)
[![Build Status](https://travis-ci.org/keisen/tf-keras-vis.svg?branch=master)](https://travis-ci.org/keisen/tf-keras-vis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

tf-keras-vis is a visualization toolkit for debugging Keras models with Tensorflow2, but not original Keras.

The features of tf-keras-vis are based on [keras-vis](https://github.com/raghakot/keras-vis), but tf-keras-vis's APIs doesn't have compatibility with keras-vis's because, instead of getting it, we prioritized to get following features.

- Support processing multipul images at a time as a batch
- Support tf.keras.Model that has multipul inputs (and, of course, multipul outpus too)
- Allow to use optimizers that embeded in tf.keras

And then we will add some algorisms such as below.

- [SmoothGrad: removing noise by adding noise](https://arxiv.org/pdf/1706.03825.pdf) (DONE)
- Deep Dream
- Style transfer


## Requirements

* Python 3.5, 3.6, 3.7 or 3.8
* tensorflow>=2.0.0


## Installation

* PyPI

```bash
$ pip install tf-keras-vis tensorflow
```

* Docker

```bash
$ docker pull keisen/tf-keras-vis:0.2.4
```

> You can find other images (that's nvidia-docker images) at [dockerhub](https://hub.docker.com/repository/docker/keisen/tf-keras-vis/tags).


## Usage

Please see [examples/attentions.ipynb](https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb), [examples/visualize_dense_layer.ipynb](https://github.com/keisen/tf-keras-vis/blob/master/examples/visualize_dense_layer.ipynb) and [examples/visualize_conv_filters.ipynb](https://github.com/keisen/tf-keras-vis/blob/master/examples/visualize_conv_filters.ipynb).


* Run Jupyter notebooks on Docker

```bash
$ docker run -itd -v /PATH/TO/tf-keras-vis:/tf-keras-vis -p 8888:8888 keisen/tf-keras-vis:0.2.4
```

Or, if you have GPU processors,

```bash
$ docker run -itd --runtime=nvidia -v /PATH/TO/tf-keras-vis:/tf-keras-vis -p 8888:8888 keisen/tf-keras-vis:0.2.4-gpu
```


## API Documentation

T.B.D.


## Known Issues

* With InceptionV3, ActivationMaximization doesn't work well, that's, it might generate meanninglessly bulr image.
* With cascading model, Gradcam doesn't work well, that's, it might occur some error.
* Unsupported `channels-first` models and datas.
