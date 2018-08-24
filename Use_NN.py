import pylab
import os
# import numpy as np
# import pandas as pd
from scipy.misc.pilutil import imread
# import tensorflow as tf


# input_num_units = 28*28
# hidden_num_units = 500
# output_num_units = 10

# path_to_model = 'C:/Users/Daniel/Desktop/DataSet/model/model'

# To stop potential randomness
# seed = 128
# rng = np.random.RandomState(seed)


root_dir = os.path.abspath('./DataSet/MNIST/')
data_dir = os.path.join(root_dir, 'data')

# test = pd.read_csv(os.path.join(data_dir, 'mnist_test.csv'))
#
# temp = []
# for item in test.values:
#     temp.append(np.array(item[1:], dtype='float32'))
#
# test_x = np.stack(temp)
#
#
#
#
# def dense_to_one_hot(labels_dense, num_classes=10):
#     """Convert class labels from scalars to one-hot vectors"""
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#
#     return labels_one_hot
#
#
# def preproc(unclean_batch_x):
#     """Convert values to range 0-1"""
#     temp_batch = unclean_batch_x / unclean_batch_x.max()
#
#     return temp_batch
#
#
# def batch_creator(batch_size, dataset_length, dataset_name):
#     """Create batch with random samples and return appropriate format"""
#     batch_mask = rng.choice(dataset_length, batch_size)
#
#     batch_x = eval(dataset_name + '_x')[[batch_mask]]
#     batch_x = preproc(batch_x)
#
#     if dataset_name == 'train':
#         batch_y = eval(dataset_name).ix[batch_mask, 'number'].values
#         batch_y = dense_to_one_hot(batch_y)
#
#     return batch_x, batch_y
#
#
# # define placeholders
# x = tf.placeholder(tf.float32, [None, input_num_units])
# y = tf.placeholder(tf.float32, [None, output_num_units])
#
# # set remaining variables
# epochs = 5
# batch_size = 128
# learning_rate = 0.0025
#
# # define weights and biases of the neural network
#
# weights = {
#     'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
#     'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
# }
#
# biases = {
#     'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
#     'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
# }
#
#
# # Add layers
# hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
# hidden_layer = tf.nn.relu(hidden_layer)
#
# output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
#
#
# # Define cost
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))
#
#
# # Backprop algorythm, Adam is optimized gradient descent
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
#
#
# with tf.Session() as sess:
#     saver.restore(sess, path_to_model)
#
#     predict = tf.argmax(output_layer, 1)
#     pred = predict.eval({x: test_x})
#
#     while (True):
#         pic = input("Index of desired test image (0 - 10,000): ")
#         try:
#             print (pred[int(pic)])
#             img_name = pic + '.png'
#             filepath = os.path.join(root_dir, 'data', 'Images', 'test', img_name)
#
#             img = imread(filepath, flatten=True)
#
#             pylab.imshow(img, cmap='gray')
#             pylab.axis('off')
#             pylab.show()
#         except IndexError:
#             break


def test_nn(pred, pic_index):
    try:
        print("Model Prediction: {}".format(pred[int(pic_index)]))
        img_name = '{}.png'.format(pic_index)
        filepath = os.path.join(root_dir, 'data', 'Images', 'test', img_name)

        img = imread(filepath, flatten=True)

        pylab.imshow(img, cmap='gray')
        pylab.axis('off')
        pylab.show()
    except FileNotFoundError or IndexError:
        print("Invalid index")
