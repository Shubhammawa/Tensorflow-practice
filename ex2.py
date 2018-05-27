import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)

n_nodes_hl1 = 784
n_nodes_hl2 = 784
n_nodes_hl3 = 784

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

	layer_1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	layer_1 = tf.nn.sigmoid(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	layer_2 = tf.nn.sigmoid(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	layer_3 = tf.nn.sigmoid(layer_3)

	output = tf.add(tf.matmul(layer_3, output_layer['weights']), output_layer['biases'])

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 70

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(0,hm_epochs+1):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))	

train_neural_network(x)