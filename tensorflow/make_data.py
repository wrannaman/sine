import tensorflow as tf
import sys
import numpy as np
import os

# convert to a tensorflow train feature of type int 64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Create data set with num samples, in tfrecords format with name as file name
def makeDataSet(name, num_samples):
    print("*** WRITING ***")
    x = np.array([
        np.random.uniform(-np.pi, np.pi, num_samples),
        np.random.uniform(-1, 1, num_samples)
    ]).T
    y = (np.sin(x[:, 0]) < x[:, 1]).astype(np.float32) * 2 - 1
    # writing data
    filename = os.path.join( os.getcwd(), name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(num_samples):
        example = tf.train.Example(features=tf.train.Features(feature={
          'x': _float_list_feature(x[i]),
          'y': _int64_feature(int(y[i]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print(" wrote {} samples to '{}.tfrecords'".format(num_samples, name))


makeDataSet("train", 10000)
makeDataSet("test", 1000)
makeDataSet("validate", 1000)
