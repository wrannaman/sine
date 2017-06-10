# DATA
###########################################################################
import pickle
import numpy as np
import tensorflow as tf
import time
with open('/Users/andrewpierno/Desktop/personal/dev/ai/sine/kur/data/train.pkl', 'rb') as fh:
    data = pickle.loads(fh.read())

with open('/Users/andrewpierno/Desktop/personal/dev/ai/sine/kur/data/test.pkl', 'rb') as fh:
    test_data = pickle.loads(fh.read())

l = list(data.keys())
X = data['point']
Y = data['above']
y_hot = []

for i in range(len(Y)):
    if Y[i] == -1:
        y_hot.append(0)
    else:
        y_hot.append(1)

X_Test = test_data['point']
Y_Test = test_data['above']
Y_Test_Hot = [];

for i in range(len(Y_Test)):
    if Y_Test[i] == -1:
        Y_Test_Hot.append(0)
    else:
        Y_Test_Hot.append(1)

# Summaries
###########################################################################
summaries_dir = '/Users/andrewpierno/Desktop/personal/dev/ai/sine/tensorflow/board'


# MODEL
###########################################################################

# Network Parameters
n_hidden_1 = 128 # 1st layer number of features
n_input = 2 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

x = tf.placeholder("float", [None, 2], name="x")
y = tf.placeholder("float", [None, 1], name="y")

"""
Before we connect the layer, however, we'll flatten our feature map (pool2) to shape [batch_size, features], so that our tensor has only two dimensions:
"""
X_Flat = tf.reshape(x, [-1, 2])
dense1 = tf.layers.dense(inputs=X_Flat, units=128, activation=tf.nn.relu)
"""
our output layer from this has [batch_size, 1024]
"""
logits = tf.layers.dense(inputs=dense1, units=1)
"""
output of this layer has shape [batch_size, 1]

returns:
    - raw values for our predictions
"""
Y_Flat = tf.reshape(y, [-1, 1])
loss = tf.losses.mean_squared_error(Y_Flat, logits)
# train_op = None
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Accuracy

# Initializing the variables
init = tf.global_variables_initializer()

# # Logging
# tf.logging.set_verbosity(tf.logging.INFO)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # # Tensorboard
    # train_writer = tf.train.SummaryWriter(summaries_dir, sess.graph)

    # Training cycle
    for epoch in range(5):
        # Loop over all batches
        total_loss = 0
        acc = 0
        counter = 0
        start = 0
        end = 1000
        while counter < 100000:
            _x = tf.reshape(X[start:end], [-1, 2])
            _y = tf.reshape([y_hot[start:end]], [-1, 1])
            _, train_loss, train_logits = sess.run([ train_op, loss, logits ], feed_dict={ x: sess.run(_x), y: sess.run(_y)})
            total_loss += train_loss;

            counter += 1000
            end += 1000
            start += 1000

            actual = sess.run(_y)
            pred = train_logits
            for i in range(1000):
                if (actual[i] == 0 and pred[i] < 0.5):
                    acc += 1
                if (actual[i] == 1 and pred[i] > 0.5):
                    acc += 1

            if counter % 5000 == 0:
                print("epoch", epoch, "i", counter, "    _acc ", int((acc/(counter + 1)) * 100),  "Ave_loss ", train_loss )

        print("\nEND EPOCH epoch", epoch, " _acc ", int((acc/(counter + 1)) * 100),  "Ave_loss ", total_loss/100, '\n')

    print("Optimization Finished!")
