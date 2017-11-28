import numpy as np 
import tensorflow as tf 
a = np.arange(100)
data = np.array([np.roll(a, i) for i in range(100)])
data = np.reshape(data, [-1,100,1])
target = np.array([np.roll(a, i) for i in range(100)])
target = np.reshape(target, [-1,100,1])


f_graph = tf.Graph()

with f_graph.as_default():
	in_ds = tf.data.Dataset.from_tensor_slices(tf.cast(data, 
		dtype = tf.float32))
	in_ds = in_ds.batch(10)
	in_iter = in_ds.make_initializable_iterator()

	t_ds = tf.data.Dataset.from_tensor_slices(tf.cast(target, 
		dtype = tf.float32))
	t_ds = t_ds.batch(10)
	t_iter = t_ds.make_initializable_iterator()

	in_seq = in_iter.get_next()
	t_seq = t_iter.get_next()


	cell = tf.contrib.rnn.LSTMBlockCell(128, use_peephole = True)
	output, state = tf.nn.dynamic_rnn(
		cell = cell,
		inputs = in_seq,
		sequence_length = None,
		dtype = tf.float32)
	prediction = tf.layers.dense(inputs = output,
		units = 1)
	loss = tf.losses.mean_squared_error(
		labels = t_seq,
		predictions = prediction)
	optimizer = tf.train.GradientDescentOptimizer(0.001)
	update_step = optimizer.minimize(loss)
	init = tf.global_variables_initializer()
	full_saver = tf.train.Saver()

sess = tf.Session(graph = f_graph)
sess.run(init)
sess.run(in_iter.initializer)
sess.run(t_iter.initializer)

for step in range(100//10):
	l, _, pred = sess.run([loss, update_step, prediction])
	print('Step %d\tLoss %f' %(step, l))
	#if step%10==0:
	#	print(np.reshape(pred, [1,100]))
full_saver.save(sess, './prova/model')
