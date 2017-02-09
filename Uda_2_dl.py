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
print('Test set', test_dataset.shape, test_labels.shape)


#We're first going to train a multinomial logistic regression using simple gradient descent.
#TensorFlow works like this:
#First you describe the computation that you want to see performed: 
##what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph. 
##This description is all contained within the block below:
###with graph.as_default():

#Then you can run the operations on this graph as many times as you want by calling session.run()
#providing it outputs to fetch from the graph that get returned. 
##with tf.Session(graph=graph) as session:

#with gradient descent training ,even this much data is prohitive
#subset the training data for faster turnaround
train_subset = 200000

graph = tf.Graph()
with graph.as_default():
  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset,:])#this means pick the first k rows
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  
  weights = tf.Variable(
    tf.truncated_normal([image_size*image_size,num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.

  logits = tf.matmul(tf_train_dataset,weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))

  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.  
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.

  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

  #Let's run this computation and iterate:
num_steps = 801
  
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])  ##.shape[0] is num of row

with tf.Session(graph=graph) as session:
  with tf.device("/gpu:0"):
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
      # Run the computations. We tell .run() that we want to run the optimizer,
      # and get the loss value and the training predictions returned as numpy
      # arrays.
      _, l, predictions = session.run([optimizer, loss, train_prediction])##remember this
      if (step % 100 == 0):
        print('Loss at step %d: %f' % (step, l))
        print('Training accuracy: %.1f%%' % accuracy(
          predictions, train_labels[:train_subset, :]))
        # Calling .eval() on valid_prediction is basically like calling run(), but
        # just to get that one numpy array. Note that it recomputes all its graph
        # dependencies.
        print('Validation accuracy: %.1f%%' % accuracy(
          valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

#Let's now switch to stochastic gradient descent training instead, which is much faster.
#The graph will be similar, except that instead of holding all the training data into a constant node, 
#we create a Placeholder node which will be fed actual data at every call of session.run().
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

  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

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
















  

















