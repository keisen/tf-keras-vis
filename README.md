# tf-keras-vis
[![Downloads](https://pepy.tech/badge/tf-keras-vis)](https://pepy.tech/project/tf-keras-vis)
[![PyPI version](https://badge.fury.io/py/tf-keras-vis.svg)](https://badge.fury.io/py/tf-keras-vis)
[![Build Status](https://travis-ci.org/keisen/tf-keras-vis.svg?branch=master)](https://travis-ci.org/keisen/tf-keras-vis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

tf-keras-vis is a visualization toolkit for debugging `tf.keras` models in Tensorflow2.

The features of tf-keras-vis are based on [keras-vis](https://github.com/raghakot/keras-vis), but tf-keras-vis's APIs doesn't have compatibility with keras-vis's, because we prioritized to get following features for our expriments.

- Support processing multiple images at a time as a batch
- Support tf.keras.Model that has multiple inputs (and, of course, multiple outpus too)
- Allow to use optimizers that embeded in tf.keras


## Visualizations

### Visualize Dense Layer

<img src='https://github.com/keisen/tf-keras-vis/raw/master/examples/images/visualize-dense-layer.png' width='600px' />

### Visualize Convolutional Filer

<img src='https://github.com/keisen/tf-keras-vis/raw/master/examples/images/visualize-filters.png' width='800px' />

### Saliency Map and GradCAM

<img src='https://github.com/keisen/tf-keras-vis/raw/master/examples/images/gradcam.png' width='400px' />


## Requirements

* Python 3.5-3.8
* tensorflow>=2.0


## Installation

* PyPI

```bash
$ pip install tf-keras-vis tensorflow
```

* Docker (container that run Jupyter Notebook)

```bash
$ docker run -itd -p 8888:8888 keisen/tf-keras-vis:0.2.4
```

If you have GPU processors,

```bash
$ docker run -itd --runtime=nvidia -p 8888:8888 keisen/tf-keras-vis:0.2.4-gpu
```

> You can find other images at [Docker Hub](https://hub.docker.com/repository/docker/keisen/tf-keras-vis/tags).


## Usage

Please see [examples/attentions.ipynb](https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb), [examples/visualize_dense_layer.ipynb](https://github.com/keisen/tf-keras-vis/blob/master/examples/visualize_dense_layer.ipynb) and [examples/visualize_conv_filters.ipynb](https://github.com/keisen/tf-keras-vis/blob/master/examples/visualize_conv_filters.ipynb) for details.


## Known Issues

* With InceptionV3, ActivationMaximization doesn't work well, that's, it might generate meanninglessly bulr image.
* With cascading model, Gradcam doesn't work well, that's, it might occur some error.
* Unsupported `channels-first` models and datas.


## ToDo
* API documentations
* We're going to add some algorisms such as below.
   - [GradCAM++](https://arxiv.org/abs/1710.11063)
   - [SmoothGrad: removing noise by adding noise](https://arxiv.org/pdf/1706.03825.pdf) (DONE)
   - Deep Dream
   - Style transfer

