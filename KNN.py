import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
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

def knn(path):
    #data manipulation starts
    whole_data, idx2filename = get_data(path)

    x_data, y_labels = featuring(whole_data) #x_data = [mean_features, std_features, skew_features, median_features, final_acc_matrix, final_lin_matrix]

    x_data = concatenate_data(x_data)

    train_x_data, train_y_data, test_x_data, test_y_data = train_test_divide(x_data, y_labels)

    #hyper-parameter 'n_neighbors' test

    n_neighbors_accuracy = {}
    for n_neighbors in range(1, 10):
        kclassifier = KNeighborsClassifier(n_neighbors = n_neighbors)
        kclassifier.fit(train_x_data, train_y_data)
        y_pred = kclassifier.predict(test_x_data)
        prediction_result = [int(result) for result in y_pred == test_y_data]
        accuracy = np.mean(prediction_result)
        n_neighbors_accuracy.update({n_neighbors:accuracy})

    return n_neighbors_accuracy

avg_accuracy = []
path = "data/"

for _ in range(100):
    accuracy = knn(path)
    avg_accuracy.append(list(accuracy.values()))

print(np.mean(np.array(avg_accuracy), axis = 0))
