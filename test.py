import tensorflow as tf

#tf.enable_eager_execution()

print(tf.executing_eagerly())       # => True

tf_variable = tf.Variable([[0.1]], tf.float32)
random = tf.random_normal([1,1])
add = tf.assign(tf_variable, tf.add(tf_variable, random))
print(add)

#constant = tf.constant([1, 2, 3])
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	for _ in range(3):
		print(sess.run(add))
		print(sess.run(tf_variable))
	#print(constant.eval())

x = [[2.]]
m = tf.matmul(x, x)
#print("hello, {}".format(m.eval()))