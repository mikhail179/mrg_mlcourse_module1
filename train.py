import numpy as np
import struct as st
from sklearn import metrics
from sklearn.metrics import classification_report
import copy
import time
import pickle
import argparse



#Signum
def sign(x):
    x[x >= 0.0] = 1
    x[x < 0] = -1
    return x

#Weight initialization
def initialize_weights(X):
    return -1 + 2 * np.random.rand(X.shape[1], 1)

#Gradient in point
def gradient(x, y, w, C):
    if y * np.dot(x.T, w) > 1:
        return w
    else:
        return w - C * y * x

#SGD
def stoh_grad_descent(X, y, w, epochs, C=100, lr=0.00001, number=0):
    n = X.shape[0]
    best = 0
    best_w = copy.copy(w)
    start = time.time()
    for i in range(epochs):
        for j in range(n):
            w = w - lr * gradient(X[j], y[j], w, C)
        if i % (epochs // 10) == 0:
            tmp = metrics.f1_score(labels[:, number], predict(train_images, w))
            if  tmp >= best:
                best = tmp
                best_w = copy.copy(w)
            end = time.time()
            print('{} epoch, f1 score:'.format(i), tmp, 'time:', end - start)
            start = end    
            tmp = metrics.f1_score(labels[:, number], predict(train_images, w))
    if  tmp >= best:
        best_w = copy.copy(w)
    return best_w


#Binary classifier prediction
def predict(X, w):
    X = X.reshape(X.shape[0], X.shape[1])
    score = np.dot(X, w)
    pred_label = sign(score)
    
    return pred_label

#One-vs-all classifier prediction
def final_prediction(X, w):
    X = X.reshape(X.shape[0], X.shape[1])
    predictions = np.empty((X.shape[0], 10))
    for i in range(predictions.shape[1]):
        predictions[:, i] = np.dot(X, w[i]).ravel()
    tmp = np.arange(10)
    return tmp[predictions.argmax(axis=1)]

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--x_train_dir', default='data/train-images.idx3-ubyte')
parser.add_argument('--y_train_dir', default='data/train-labels.idx1-ubyte')
parser.add_argument('--model_output_dir', default='model/weights.pkl')
args = parser.parse_args()


#Reading the data
with open(args.y_train_dir, 'rb') as flbl:
    magic, num = st.unpack(">II", flbl.read(8))
    train_labels = np.fromfile(flbl, dtype=np.int8)

with open(args.x_train_dir, 'rb') as fimg:
    magic, num, rows, cols = st.unpack(">IIII", fimg.read(16))
    train_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_labels), rows * cols)
 
#Transforming, adding bias and scaling the data
train_images = train_images / 255
train_images = np.hstack((train_images, train_images ** 2))
train_images = np.hstack((train_images, np.ones((train_images.shape[0], 1))))
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], 1)
#train_images[train_images > 0] = 1


#Preparing data for one-vs-all
train_labels = train_labels.reshape(train_labels.shape[0], 1)
labels = np.apply_along_axis(lambda x: (x == 0) * 1, arr=train_labels, axis=0)
labels[labels == 0] = -1
for i in range(1, 10):
    tmp = np.apply_along_axis(lambda x: (x == i) * 1, arr=train_labels, axis=0)
    tmp[tmp == 0] = -1
    labels = np.hstack((labels, tmp))

#Initializing weights for binary classifiers
w = initialize_weights(train_images)

#Initializing weight matrix for one-vs-all
weights = np.empty((10, w.shape[0], w.shape[1]))

#Running 10 binary classifiers
#for 100 epochs, training time is ~8, f1 score = 0.94 (train), f1 score = 0.93 (predict)
for i in range(10):
    print('{} classifier'.format(i))
    weights[i] = stoh_grad_descent(train_images, labels[:, i], w, epochs=100, C=10000, lr = 0.0000001, number=i)
    
pkl_filename = args.model_output_dir
with open(pkl_filename, 'wb') as file:  
    pickle.dump(weights, file)

predicted_train = final_prediction(train_images, weights)
print(classification_report(train_labels.ravel(), predicted_train))