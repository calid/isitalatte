## Description

This repo contains code for generating a binary image classifier that can
identify images of lattes.  It does this using
[TensorFlow](https://www.tensorflow.org/), [ImageNet](http://image-net.org/),
and [Inception v3](https://arxiv.org/abs/1512.00567).

## Install Dependencies

The code was written using Python 3.6.6.  You can install this (and other)
versions of python easily using [pyenv](https://github.com/pyenv/pyenv).

```
$ git clone https://github.com/calid/isitalatte.git
$ cd isitalatte
$ pip3 install -r requirements.txt
```

You will also need the data used to train the classifier (images from ImageNet,
model weights from Inception v3).  You can download these
[here](https://s3.amazonaws.com/isitalatte/isitalatte-resources.tar.gz). Simply
extract in the repo directory and you're all set.

## Train the Model

`python3 train_isitalatte.py`

This can take 10-15 minutes depending on your machine.  Once finished the
model will be saved to `isitalatte.h5`.  The training code will also generate
`accuracy_and_loss.png`, a chart of the training/validation accuracy and loss
over time.

## Evaluate the Model

`python3 eval_isitalatte.py`

This will use TensorFlow's builtin evaluate mechanism to test the model
against an additional data set not used during training and print the
corresponding accuracy and loss for this data set.

## Run the Model Against Arbitrary Images

A command line script is provided to feed images into the model for ad-hoc
predictions. Run it as:

`./bin/isitalatte </path/to/image/file>`

## See Also

https://github.com/calid/isitalatte-app, code for isitalatte web frontend
