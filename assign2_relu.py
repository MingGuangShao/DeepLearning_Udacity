# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

#Load the data we generated ----notMNIST.pickle

pickle_file = '/media/dat1/shao/udacity/notMNIST.pickle'

with open(pickle_file,'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save #hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


##next reformat into a shape that's more adapted to the models we're going to tain:
#data as a float matrix
#labels as float 1-hot encodings

image_size = 28
num_labels = 10

def reformat(dataset,labels):
  #learn the mothod to reshape the data for tf
  dataset = dataset.reshape((-1,image_size*image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  #example  np.arange(10) == 1
  #array([False,  True, False, False, False, False, False, False, False, False], dtype=bool)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset,labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

batch_size = 128
graph = tf.Graph()
with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape = (batch_size,image_size*image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  hidden_node_count = 1024
  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_node_count]))
  biases1 = tf.Variable(tf.zeros([hidden_node_count]))
 
  weights2 = tf.Variable(
    tf.truncated_normal([hidden_node_count,num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))

  # Training computation.#######RELU
  ys = tf.matmul(tf_train_dataset,weights1) + biases1
  hidden = tf.nn.relu(ys)
  logits = tf.matmul(hidden, weights2) + biases2
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,weights1) + biases1), weights2) + biases2)
  test_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,weights1) + biases1), weights2) + biases2)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])  ##.shape[0] is num of row
##Let's run it:

num_steps = 30000

with tf.Session(graph=graph) as session:
  with tf.device("/gpu:0"):
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 500 == 0):
        print("Minibatch loss at step %d: %f" % (step, l))
        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        print("Validation accuracy: %.1f%%" % accuracy(
          valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))





