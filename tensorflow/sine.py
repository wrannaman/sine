import tensorflow as tf

# Find out which devices your operatrion and tensors are assigned to.
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
