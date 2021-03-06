import os
import numpy as np
# import pandas as pd
# from scipy.misc.pilutil import imread
# import pylab
import tensorflow as tf
import scipy.io

USE_SAVED = False
SAVE_ON_EXIT = True

input_num_units = 28*28
hidden1_num_units = 600
hidden2_num_units = 350
output_num_units = 62

path_to_model = 'C:/Users/Daniel/Desktop/DataSet/EMNIST/model2/model'

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


root_dir = os.path.abspath('C:/Users/Daniel/Desktop/DataSet/EMNIST/')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

data = scipy.io.loadmat(os.path.join(root_dir, 'matlab', 'emnist-byclass.mat'))['dataset']

# Training and validation sets are already separately defined
train_x, train_y, _ = data[0,0][0][0,0]
val_x, val_y, _ = data[0,0][1][0,0]

# Character map for determining correct character
maps = data[0,0][2]
char_map = {item[0] : chr(item[1]) for item in maps}

# temp = []
# for item in train.values:
#     temp.append(item[0])
#     item = item[1:]
# train.label = temp

# print (train.head())
# print (train.number)


#
# # sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

# train['filename'] = pd.Series([str(i) + '.png' for i in range(len(train.values))])
# test['filename'] = pd.Series([str(i) + '.png' for i in range(len(test.values))])

# print(train.head())

# print(train.columns)
# print(train.values)

# print(len(train.values[0]))


# r = np.random.randint(0,10000)
# print (train.values[r][0])
# print(r)
# img_name = '{}.png'.format(r)
# filepath = os.path.join(data_dir, 'Images', 'train', img_name)
#
# img = imread(filepath, flatten=True)
#
# pylab.imshow(img, cmap='gray')
# pylab.axis('off')
# pylab.show()



def dense_to_one_hot(labels_dense, num_classes=62):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()

    return temp_batch


def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = eval(dataset_name + '_x')[[batch_mask]]
    batch_x = preproc(batch_x)

    if dataset_name == 'train':
        batch_y = eval(dataset_name + '_y')[[batch_mask]]
        batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y



# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 3
batch_size = 512
learning_rate = 0.01

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden1_num_units], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden1_num_units, hidden2_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden2_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden1_num_units], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden2_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}


# Add layers
hidden1_layer = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden1_layer = tf.nn.relu(hidden1_layer)       # Layer 1 activation function

hidden2_layer = tf.add(tf.matmul(hidden1_layer, weights['hidden2']), biases['hidden2'])
hidden2_layer = tf.nn.relu(hidden2_layer)       # Layer 2 activation function

output_layer = tf.matmul(hidden2_layer, weights['output']) + biases['output']


# Define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))


# Backprop algorythm, Adam is optimized gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    # Attempt to load saved model, otherwise create initialized variables
    if (USE_SAVED):
        try:
            saver.restore(sess, path_to_model)
        except ValueError:
            sess.run(init)
    else:
        sess.run(init)

    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize

    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train_x.shape[0] / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        print ("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

    print ("\nTraining complete!")

    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

    print ("Validation Accuracy:", accuracy.eval({x: val_x, y: dense_to_one_hot(val_y)}))

    # predict = tf.argmax(output_layer, 1)
    # pred = predict.eval({x: test_x})

    # from Use_NN import test_nn
    # while (True):
    #     pic = input("Index of picture in test set (Enter -1 to exit): ")
    #     if int(pic) == -1:
    #         break
    #     else:
    #         test_nn(pred, int(pic))

    if SAVE_ON_EXIT:
        save_path = saver.save(sess, path_to_model)
        print ('Model saved to: {}'.format(save_path))