from PIL import Image
import numpy as np
# import pylab
import os
import pandas as pd
# from scipy.misc.pilutil import imread


root_dir = os.path.abspath('./DataSet/MNIST/')
data_dir = os.path.join(root_dir, 'data')
# sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
# os.path.exists(sub_dir)

train = pd.read_csv(os.path.join(data_dir, 'mnist_train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'mnist_test.csv'))



print("Train")

for item, num in zip(train.values, range(len(train.values))):
    if not num % 1000:
        print(num)
    img = np.empty((28,28), np.uint8)
    img.shape = 28, 28

    for i in range(28 * 28):
        img[27- (i // 28), i % 28] = item[i + 1]


    image = Image.frombuffer('L', (28,28), img)
    image.save(os.path.join(data_dir, 'Images', 'train', '{}.png'.format(num)))

print("Test")

for item, num in zip(test.values, range(len(test.values))):
    if not num % 1000:
        print(num)
    img = np.empty((28,28), np.uint8)
    img.shape = 28, 28

    for i in range(28 * 28):
        img[27- (i // 28), i % 28] = item[i + 1]


    image = Image.frombuffer('L', (28,28), img)
    image.save(os.path.join(data_dir, 'Images', 'test', '{}.png'.format(num)))

