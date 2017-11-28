import numpy as np 
import tensorflow as tf 

a = np.arange(100)
#a = np.flip(a, axis = 0)
data = np.array([np.roll(a, i) for i in range(100)])
data = np.reshape(data, [-1,100,1])

prova = np.arange(128)

restored_graph = tf.Graph()

with restored_graph.as_default():
	val_ds = tf.data.Dataset.from_tensor_slices(tf.cast(data, 
		dtype = tf.float32))
	val_ds = val_ds.batch(1)
	val_iter = val_ds.make_initializable_iterator()


	val_seq = val_iter.get_next()


	cell = tf.contrib.rnn.LSTMBlockCell(128, use_peephole = True)
	output, state = tf.nn.dynamic_rnn(
		cell = cell,
		inputs = val_seq,
		sequence_length = None,
		dtype = tf.float32)
	prediction = tf.layers.dense(inputs = output,
		units = 1)
	saver = tf.train.Saver()
	p = tf.constant(prova, dtype = tf.float32)
	p = tf.reshape(p, [1,128])
	h = tf.constant(np.flip(prova,0), dtype = tf.float32)  
	h = tf.reshape(h, [128,1])
	z = tf.matmul(h, p)

restored_session = tf.Session(graph = restored_graph)
saver.restore(restored_session, './prova/model')
restored_session.run(val_iter.initializer)
for step in range(100):
	val, pred, Z = restored_session.run([val_seq, prediction, z])
	print(np.reshape(val, [1,100]))
	print(np.reshape(pred, [1,100]))
	print(Z)