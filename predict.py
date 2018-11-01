import numpy as np
import struct as st
from sklearn import metrics
from sklearn.metrics import classification_report
import copy
import time
import pickle
import argparse


def final_prediction(X, w):
    X = X.reshape(X.shape[0], X.shape[1])
    predictions = np.empty((X.shape[0], 10))
    for i in range(predictions.shape[1]):
        predictions[:, i] = np.dot(X, w[i]).ravel()
    tmp = np.arange(10)
    return tmp[predictions.argmax(axis=1)]

parser = argparse.ArgumentParser()
parser.add_argument('--x_test_dir', default='data/t10k-images.idx3-ubyte')
parser.add_argument('--y_test_dir', default='data/t10k-labels.idx1-ubyte')
parser.add_argument('--model_output_dir', default='model/weights.pkl')
args = parser.parse_args()

with open(args.y_test_dir, 'rb') as flbl:
    magic, num = st.unpack(">II", flbl.read(8))
    test_labels = np.fromfile(flbl, dtype=np.int8)

with open(args.x_test_dir, 'rb') as fimg:
    magic, num, rows, cols = st.unpack(">IIII", fimg.read(16))
    test_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_labels), rows * cols)
    
test_images = test_images / 255
test_images = np.hstack((test_images, test_images ** 2))
test_images = np.hstack((test_images, np.ones((test_images.shape[0], 1))))
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], 1)


pkl_filename = args.model_output_dir
with open(pkl_filename, 'rb') as file:  
    weights = pickle.load(file)

predicted_test = final_prediction(test_images, weights)
print(classification_report(test_labels.ravel(), predicted_test))
