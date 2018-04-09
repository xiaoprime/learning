import tensorflow as tf
import os

TENSORBOARD_FOLDER = os.getcwd()+'/__tensorboard__'
#tf.enable_eager_execution()

print(tf.executing_eagerly())

tf_variable = tf.Variable([[0.1]], tf.float32)
random = tf.random_normal([1,1])
add = tf.assign(tf_variable, tf.add(tf_variable, random))
print(add)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	for _ in range(3):
		sess.run(add)
		print(tf_variable.eval())
	train_writer = tf.summary.FileWriter(TENSORBOARD_FOLDER, sess.graph)
	train_writer.close()