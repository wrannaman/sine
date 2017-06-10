# Init
###########################################################################
import tensorflow as tf
import os
import argparse
import time
import sys

FLAGS = None
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validate.tfrecords'

# Read and decode a single example from a TF Record
###########################################################################
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'x': tf.FixedLenFeature([2], tf.float32),
            'y': tf.FixedLenFeature([], tf.int64)
        }
    )
    return features['x'], features['y']

# TF Magic to get examples of size batch_size
###########################################################################
def inputs(train, batch_size, num_epochs):
    """
        Reads input data num_epoch times.

        Args:
            train: bool. training = true else validation = true
            batch_size: num examples per batch size
            num_epochs: # times to read input data or none to train forever

        Returns:
            A tuple (x, y) where x is an array of lenth 2 with a point and y is -1 or 1
    """
    filename = os.path.join(os.getcwd(), TRAIN_FILE if train else VALIDATION_FILE)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        x, y = read_and_decode(filename_queue)

        """
            X, Y will have shape (FLAGS.batch_size, #)
        """
        X, Y = tf.train.shuffle_batch(
            [x, y], batch_size=batch_size, num_threads=10,
            capacity=1000 + 3 * batch_size,
            # ensure minimum amount of shuffling
            min_after_dequeue=1000
        )
        return X, Y

# Training
###########################################################################
def run_training():
    """
        Training for Sine wave
    """
    with tf.Graph().as_default():
        # Input X and Y coords
        X, Y = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

        dense1 = tf.layers.dense(inputs=X, units=128, activation=tf.nn.relu)

        """
            our output layer from dense1 has [batch_size, 1024]
        """
        logits = tf.layers.dense(inputs=dense1, units=1)

        """
            output of this logits layer has shape [batch_size, 1]

            returns:
                - raw values for our predictions
        """
        loss = tf.losses.mean_squared_error(tf.reshape(Y, [-1, 1]), logits)

        train_op = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

        #the op for initializing variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create TF session
        sess = tf.Session()
        sess.run(init_op)

        # start input enqueue threads
        # Must be run after initialization. This is not symbolic call, it creates threads.
        # If we dont tell tensorflow to start these threads, the queue will be blocked forever
        # Waiting for data to be enqueued.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            total_loss = 0
            acc = 0
            while not coord.should_stop():
                # print("setp ", step)
                start_time = time.time()
                """
                    Run a step of the model

                    If you don't do a sess.run on the data, I don't think the file reader gets the next data record.

                    You don't need a feed_dict here because our X and Y are already tensors declared in the graph.

                    returns:
                      training loss, the y_hats and the actual y. Used to calculate accuracy and loss.
                """
                _, train_loss, predicted, actual = sess.run([ train_op, loss, logits, Y ])

                # Accuracy and Loss calculations
                total_loss += train_loss;
                for i in range(FLAGS.batch_size):
                    if actual[i] == 0 and predicted[i][0] < .5:
                        acc += 1
                    if actual[i] == 1 and predicted[i][0] >= .5:
                        acc += 1
                duration = time.time() - start_time
                print("Step {}, ({} sec)({} loss)({} acc)".format(step, duration, train_loss, (acc / ( (step if step > 0 else 1) * FLAGS.batch_size ))))
                step += 1

        except tf.errors.OutOfRangeError:
            saver = tf.train.Saver()
            saver.save(sess, 'saved/sine-0.0.1')
            saver.export_meta_graph('saved/sine-0.0.1.meta')
            print("Done training for %d epochs, %d steps. %f Total" % (FLAGS.num_epochs, step, (FLAGS.num_epochs * (step)) ))
        finally:
            # ask threads to stop
            coord.request_stop()

        # wait for threads to finish
        coord.join(threads)
        # close the session
        sess.close()

def main(_):
    run_training()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=100,
      help='Number of epochs to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1000,
      help='Batch size.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
