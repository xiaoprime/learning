import tensorflow as tf
import numpy as np
import os

TENSORBOARD_FOLDER = os.getcwd()+'/__tensorboard__'

# Load the training data into two NumPy arrays, for example using `np.load()`.
# Use fake data
#with np.load("/var/data/training_data.npy") as data:
features = np.array([1,2,3])
labels = np.array([2,3,4])

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
# dataset = ...
iterator = dataset.make_initializable_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:
	sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
	print(dataset)
	while True:
		try:
  			print(sess.run(next_element))
		except tf.errors.OutOfRangeError:
			print("End of dataset")  # ==> "End of dataset"
			break
	train_writer = tf.summary.FileWriter(TENSORBOARD_FOLDER, sess.graph)
	train_writer.close()