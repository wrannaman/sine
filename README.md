# Sine

Predicting whether a given point is above or below a sine wave taken from [Kur](http://kur.deepgram.com/tutorial.html)

There are three implementations: Kur, Keras, Tensorflow.

I started with Kur because it was the simplest, then moved down to Keras, and finally to Tensorflow.

## Goals

To learn Tensorflow and get familiar with the TFRecord batching as inputs to a model. The problem this model solves is contrived and
generally garbage, but it's small enough to know what each line of code is doing.

![Sine Wave](http://kur.deepgram.com/_images/tutorial-plot-results.png)

## To run Kur
 Take a look at the README in kur/README.md

## To run Keras
 `cd keras && python3 keras.sine.py`

## To run Tensorflow
 ### Read data from TFRecords
  `cd tensorflow && python3 train-fully-connected.py`
 ### Read data from PKL files
  `cd tensorflow && python3 train-simple.py`

### Tensorflow
 There are two implementations. One is with TFRecords and one is with .pkl files.



## Results

|  Framework  |  Accuracy | Loss  |
|-------------|-----------|-------|
| Kur         |  99.8     |       |
| Keras       |  98.7     | 0.016 |
| Tensorflow  |  93.5     | 0.026 |


The goal was to build the same model in kur, keras, and tensorflow. Obviously with such a huge drop in accuracy for Tensorflow, something
wen't awry. That problem is for another day. For now, just getting more familiar with reading and creating a tensorflow model that
runs at all is sufficient.
