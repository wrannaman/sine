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
# MODEL
###########################################################################
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras.callbacks import TensorBoard

# Tensorboard
###########################################################################
tbCallBack = TensorBoard(log_dir='board', histogram_freq=0, write_graph=True, write_images=True)

# Model
###########################################################################
model = Sequential([
    Dense(128, input_shape=(2,)),
    Activation('relu'),
    Dense(1),
    Activation('relu'),
])

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Export model as png
plot_model(model, to_file='model.png')

# Train
###########################################################################
model.fit(X, y_hot, epochs=25, batch_size=1000, callbacks=[tbCallBack])

# Eval
###########################################################################
loss_and_metrics = model.evaluate(X_Test, Y_Test_Hot, batch_size=128)

print("\nloss and metrics", loss_and_metrics)
