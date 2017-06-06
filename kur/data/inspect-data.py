import pickle
import numpy
with open('train.pkl', 'rb') as fh:
    data = pickle.loads(fh.read())

print("data", data)
print("above's shape ", data['above'].shape)
print("point's shape ", data['point'].shape)
l = list(data.keys())
x = data['point'][:10]
y = data['above'][:10]

print("list {}".format(l))
print("\nx ", x)
print("\ny", y)
