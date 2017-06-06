import tensorflow as tf
import os
import argparse
import time
import sys
# Flags are batch_size and num_epochs
FLAGS = None
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validate.tfrecords'

# Parameters
learning_rate = 0.01

# Network Params
n_hidden_1 = 128
n_input = 2
n_classes = 2


def read_and_decode(filename_queue):
    print("*** READING ***")
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
    # File is either training or validate
    filename = os.path.join(os.getcwd(), TRAIN_FILE if train else VALIDATION_FILE)
    print("filename", filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        x, y = read_and_decode(filename_queue)

        X, Y = tf.train.shuffle_batch(
            [x, y], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # ensure minimum amount of shuffling
            min_after_dequeue=1000
        )
        return x, y

def run_training():
    """
        Training for Sine wave
    """
    with tf.Graph().as_default():
        # X shape is (2, ), Y shape is ()
        X, Y = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        print("X ", X)
        print("Y ", Y)
        # Now Y is shape (1, )
        Y = tf.reshape(Y, [ 1, 1])
        X = tf.reshape(X, [ ])
        print("After ")
        print("X ", X)
        print("Y ", Y)
        # print("x shape ", tf.shape(X))
        # print("Y shape ", tf.shape(Y))
        #
        #
        def mlp(x, weights, biases):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
            return out_layer
        # # Graph input
        x = tf.placeholder("float", [ 2, None ])
        y = tf.placeholder("float", [ None ])

        # Store layers weight & bias
        weights = {
          'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
          'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        }
        biases = {
          'b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'out': tf.Variable(tf.random_normal([n_classes]))
        }
        #
        #
        # # Build the Graph
        #
        # # model:
        # #   - input: point
        # #   - dense: 128
        # #   - activation: tanh
        # #   - dense: 1
        # #   - activation: tanh
        # #     name: above
        #
        pred = mlp(x, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


        #
        # loss:
        #   - target: above
        #     name: mean_squared_error

        loss = tf.reduce_mean(tf.squared_difference(x, y))


        #the op for initializing variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create TF session
        sess = tf.Session()
        sess.run(init_op)

        # start input enqueue threads
        # Must be run after initializationi. This is not symbolic call, it creates threads.
        # If we dont tell tensorflow to start these threads, the queue will be blocked forever
        # Waiting for data to be enqueued.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # print("X shape is ", sess.run(X).shape)
        # return

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                ## @TODO
                ## Run a step of the model
                # If you don't do a sess.run, I don't think the file reader gets the next data record.
                # print("index is ", sess.run(X).shape)
                # print("index is ", sess.run(Y).shape)

                _, c = sess.run([optimizer, cost], feed_dict={
                      x: sess.run(X),
                      y: sess.run(Y)
                    })

                duration = time.time() - start_time
                if step % 100 == 0:
                    print("Step {}, ({} sec)".format(step, duration))
                    print("coord.should_stop()", coord.should_stop())
                    print("Batch ", FLAGS.batch_size, " - ", FLAGS.num_epochs)

                step += 1

        except tf.errors.OutOfRangeError:
            print("Done training for %d epochs, %d steps." % (FLAGS.num_epochs, step))
        finally:
            # ask threads to stop
            coord.request_stop()

        # wait for threads to finish
        coord.join(threads)
        # close the session
        sess.close()

        # X, Y = s.run([x, y])
        # print("X ", X)
        # print("Y ", Y)
        # X, Y = s.run([x, y])

def main(_):
    run_training()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=2,
      help='Number of epochs to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1000,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/data',
      help='Directory with the training data.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
