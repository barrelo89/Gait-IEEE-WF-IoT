import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from scipy.stats import skew

def get_data(path):
    columns = ['accx', 'accy', 'accz', 'linx', 'liny', 'linz']

    idx2filename = {}

    whole_data = []

    for index, file in enumerate(os.listdir(path)):
        data = pd.read_csv(path + str(file), names=columns, delimiter=',')
        idx2filename[index] = file
        whole_data.append(data.values[1000:4000, :])

    return whole_data, idx2filename

def featuring(datas):  # take mean and std of data samples and plus RMS

    mean_features = []
    std_features = []
    skew_features = []
    median_features = []
    final_acc_matrix = []
    final_lin_matrix = []

    y_labels = []

    for idx, data in enumerate(datas):

        one_data_size, num_features = data.shape

        num_sample = 30
        one_sample_size = int(one_data_size / 30)
        for num in range(num_sample):

            mean_features.append(np.mean(data[num * one_sample_size:(num + 1) * one_sample_size, :], 0))
            std_features.append(np.std(data[num * one_sample_size:(num + 1) * one_sample_size, :], 0))
            skew_features.append(skew(data[num * one_sample_size:(num + 1) * one_sample_size, :], axis=0, bias=True))
            median_features.append(np.median(data[num * one_sample_size:(num + 1) * one_sample_size, :], axis = 0))
            y_labels.append(idx)

            square_matrix = np.square(data[num * one_sample_size:(num + 1) * one_sample_size, :])
            acc_square_matrix = square_matrix[:, [0, 1, 2]]
            lin_square_matrix = square_matrix[:, [3, 4, 5]]

            acc_square_matrix = np.mean(np.sum(acc_square_matrix, axis = 1))
            lin_square_matrix = np.mean(np.sum(lin_square_matrix, axis = 1))

            sqrt_acc_features = np.sqrt(acc_square_matrix)
            sqrt_lin_features = np.sqrt(lin_square_matrix)

            final_acc_matrix.append(sqrt_acc_features)
            final_lin_matrix.append(sqrt_lin_features)

    return [np.array(mean_features), np.array(std_features), np.array(skew_features), np.array(median_features), np.array(final_acc_matrix).reshape(-1, 1), np.array(final_lin_matrix).reshape(-1, 1)], np.array(y_labels)

def concatenate_data(data):

    result = data[0]

    for idx in range(1, len(data)):
        result = np.concatenate((result, data[idx]), axis = 1)

    return result

def train_test_divide(x_data, y_data, ratio = 0.7):#mean_data, std_data, skew_data, median_data, amp_acc, amp_lin
    num_sample, num_feature = x_data.shape

    x_data, y_data = shuffle(x_data, y_data)

    train_size = int(num_sample*ratio)
    train_x_data = x_data[:train_size, :]
    test_x_data = x_data[train_size:, :]

    train_y_data = y_data[:train_size]
    test_y_data = y_data[train_size:]

    return train_x_data, train_y_data, test_x_data, test_y_data

def DNN(path):
    #data manipulation starts

    whole_data, idx2filename = get_data(path)

    x_data, y_labels = featuring(whole_data) #x_data = [mean_features, std_features, skew_features, median_features, final_acc_matrix, final_lin_matrix]
    num_class = len(set(y_labels))

    x_data = concatenate_data(x_data)

    #data_shape = (?, 26), (?, 1)
    train_x_data, train_y_data, test_x_data, test_y_data = train_test_divide(x_data, y_labels)

    _, num_feature = train_x_data.shape
    l1_dim = 128
    l2_dim = 256
    #l3_dim = 256
    initial_learning_rate = 0.001
    beta = 0.003

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = [None, num_feature])
    Y = tf.placeholder(tf.int32, shape = [None])
    Y_one_hot = tf.one_hot(Y, num_class)
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.get_variable('W1', shape = [num_feature, l1_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([l1_dim]), name = 'b1')
    L1 = tf.matmul(X, W1) + b1
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

    W2 = tf.get_variable('W2', shape = [l1_dim, l2_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([l2_dim]), name = 'b2')
    L2 = tf.matmul(L1, W2) + b2
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

    #W3 = tf.get_variable('W3', shape = [l2_dim, l3_dim], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    #b3 = tf.Variable(tf.random_normal([l3_dim]), name = 'b3')
    #L3 = tf.matmul(L2, W3) + b3
    #L3 = tf.nn.relu(L3)
    #L3 = tf.nn.dropout(L3, keep_prob = keep_prob)

    W3 = tf.get_variable('W4', shape = [l2_dim, num_class], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([num_class]), name = 'b4')
    L3 = tf.matmul(L2, W3) + b3

    hypothesis = tf.nn.softmax(L3)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = L3, labels = Y_one_hot))
    rg_cost = beta*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)+ tf.nn.l2_loss(W3))# + tf.nn.l2_loss(W4)
    loss = cost + rg_cost

    global_step = tf.Variable(0) #count the # of steps starting from 0
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 100000, 0.96)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_steps = 1000

    for step in range(num_steps):
        l, a, _ = sess.run([loss, accuracy, optimizer], feed_dict = {X: train_x_data, Y: train_y_data, keep_prob: 0.7})
        if step % 100 == 0:
            print('iteration: %d, cost: %f, accuracy: %f' %(step, l, a))

    #print(sess.run(accuracy, feed_dict = {X: test_x_data, Y: test_y_data, keep_prob: 1.0}))

    acc = sess.run(accuracy, feed_dict = {X: test_x_data, Y: test_y_data, keep_prob: 1.0})
    sess.close()

    return acc

num_iteration = 10
path = "data/"

accuracy = []
for _ in range(num_iteration):
    acc = DNN(path)
    accuracy.append(acc)

accuracy = np.array(accuracy)
print(accuracy)
print(accuracy.mean())
print(accuracy.max())
print(accuracy.min())




















#end
