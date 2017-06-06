import sys
import pickle
import numpy as np

if len(sys.argv) != 3:
    print("usage, {} NUM SAMPLES OUTPUT-FILE".format(sys.argv[0]), file=sys.stderr)
    sys.exit(1)

_, num_samples, output_file = sys.argv
num_samples = int(num_samples)

x = np.array([ 
    np.random.uniform(-np.pi, np.pi, num_samples),
    np.random.uniform(-1, 1, num_samples)
]).T
y = (np.sin(x[:, 0]) < x[:, 1]).astype(np.float32) * 2 - 1

with open(output_file, 'wb') as fh:
    fh.write(pickle.dumps({ 'point': x, 'above': y }))
