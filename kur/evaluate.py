from kur.loggers import BinaryLogger
import matplotlib.pyplot as plt
import pickle
import numpy as np

training_loss = BinaryLogger.load_column('sine-log', 'training_loss_total')
validation_loss = BinaryLogger.load_column('sine-log', 'validation_loss_total')

# Plot the losses on training and validation
plt.xlabel('Epoch')
plt.ylabel('Loss')
epoch = list(range(1, 1 + len(training_loss)))
t_line, = plt.plot(epoch, training_loss, 'co-', label='Training Loss')
v_line, = plt.plot(epoch, validation_loss, 'mo-', label='Validation Loss')
plt.legend(handles=[t_line, v_line])
plt.show()



with open('data/output.pkl', 'rb') as fh:
    data = pickle.loads(fh.read())

print(list(data.keys()))

# Result is model prediction, truth is ground truth
diff = np.abs(data['truth']['above'] - data['result']['above']) < 1
correct = diff.sum()
total = len(diff)

print("accuracy ", correct / total * 100)

#Plot the accuracy

should_be_above = data['result']['above'][ data['truth']['above'] > 0 ]
should_be_below = data['result']['above'][ data['truth']['above'] < 0 ]
plt.xlabel('Model output')
plt.ylabel('Counts')
plt.xlim(-1, 1)
plt.hist(should_be_above, 20, facecolor='r', alpha=0.5, range=(-1, 1))
plt.hist(should_be_below, 20, facecolor='b', alpha=0.5, range=(-1, 1))
plt.show()
