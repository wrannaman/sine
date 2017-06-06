import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('train.pkl', 'rb') as fh:
    data = pickle.loads(fh.read())

above = data['above'] > 0
plt.xlim(-np.pi, np.pi)
plt.plot(data['point'][above, 0], data['point'][above, 1], 'ro')
plt.plot(data['point'][~above, 0], data['point'][~above, 1], 'bo')
X = np.linspace(-np.pi, np.pi)
plt.plot(X, np.sin(X), 'k', linewidth=5)
plt.show()
