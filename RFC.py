import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def get_data(path):
    columns = ['accx', 'accy', 'accz', 'linx', 'liny', 'linz']

    idx2filename = {}

    whole_data = []

    for index, file in enumerate(os.listdir(path)):
        data = pd.read_csv(path + str(file), names=columns, delimiter=',')
        idx2filename[index] = file
        whole_data.append(data.values[1000:4000, :])

    return whole_data, idx2filename

def RMS(threedata):
    print((threedata[0:1]))

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
            lin_square_matrix = square_matrix[:, [0, 1, 2]]

            acc_square_matrix = acc_square_matrix.sum(axis = 1)
            lin_square_matrix = lin_square_matrix.sum(axis = 1)

            sqrt_acc_features = np.sqrt(acc_square_matrix)
            sqrt_lin_features = np.sqrt(lin_square_matrix)

            final_acc_matrix.append(sqrt_acc_features)
            final_lin_matrix.append(sqrt_lin_features)


    return mean_features, std_features, skew_features, median_features, final_acc_matrix, final_lin_matrix, y_labels

def train_test_divide(mean_data, std_data, skew_data, median_data, amp_acc, amp_lin, y_data, ratio):
    num_data = len(mean_data)

    mean_data, std_data, skew_data, median_data, amp_acc, amp_lin, y_data = shuffle(mean_data, std_data, skew_data, median_data, amp_acc, amp_lin, y_data)

    train_mean_data = mean_data[:int(ratio * num_data)]
    train_std_data = std_data[:int(ratio * num_data)]
    train_skew_data = skew_data[:int(ratio * num_data)]
    train_median_data = median_data[:int(ratio * num_data)]
    train_amp_acc = amp_acc[:int(ratio * num_data)]
    train_amp_lin = amp_lin[:int(ratio * num_data)]

    train_label = y_data[:int(ratio * num_data)]

    test_mean_data = mean_data[int(ratio * num_data):]
    test_std_data = std_data[int(ratio * num_data):]
    test_skew_data = skew_data[int(ratio * num_data):]
    test_median_data = median_data[int(ratio * num_data):]
    test_amp_acc = amp_acc[int(ratio * num_data):]
    test_amp_lin = amp_lin[int(ratio * num_data):]

    test_label = y_data[int(ratio * num_data):]

    return train_mean_data, train_std_data, train_skew_data, train_median_data, train_amp_acc, train_amp_lin, train_label, test_mean_data, test_std_data, test_skew_data, test_median_data, test_amp_acc, test_amp_lin, test_label

def classify(mean_features, std_features, skew_features, median_features, final_acc_matrix, final_lin_matrix, y_labels):

    train_mean_data, train_std_data, train_skew_data, train_median_data, train_amp_acc, train_amp_lin, train_label, test_mean_data, test_std_data, test_skew_data, test_median_data, test_amp_acc, test_amp_lin, test_label = train_test_divide(mean_features, std_features, skew_features, median_features, final_acc_matrix, final_lin_matrix, y_labels, 0.6)

    train_mean_data = np.array(train_mean_data)
    train_std_data = np.array(train_std_data)
    train_skew_data = np.array(train_skew_data)
    train_median_data = np.array(train_median_data)
    train_amp_acc = np.array(train_amp_acc)
    train_amp_lin = np.array(train_amp_lin)

    test_mean_data = np.array(test_mean_data)
    test_std_data = np.array(test_std_data)
    test_skew_data = np.array(test_skew_data)
    test_median_data = np.array(test_median_data)
    test_amp_acc = np.array(test_amp_acc)
    test_amp_lin = np.array(test_amp_lin)

    train_data = np.concatenate((train_mean_data, train_std_data, train_skew_data, train_median_data), axis=1)#train_amp_acc, , train_amp_lin ,  , train_skew_data, train_median_data
    test_data = np.concatenate((test_mean_data, test_std_data, test_skew_data, test_median_data), axis=1)#test_amp_acc , test_amp_lin , test_median_data , test_skew_data, test_median_data

    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(train_data, train_label)
    train_score = rfc.score(train_data, train_label)
    test_score = rfc.score(test_data, test_label)
    print("rfc train score: ", train_score)
    print("rfc test score: ", test_score)

    p = rfc.predict(test_data)
    f1_score_result = f1_score(test_label, p, average=None).mean()
    print("dt F1 score: ", f1_score_result)

    return test_score, f1_score_result

path = "data/"

whole_data, idx2filename = get_data(path)
mean_features, std_features, skew_features, median_features, final_acc_matrix, final_lin_matrix, y_labels = featuring(whole_data)
num_iteration = 20

test_scores = []
f1_scores = []

for _ in range(num_iteration):
    test_score, f1_score_result = classify(mean_features, std_features, skew_features, median_features, final_acc_matrix, final_lin_matrix, y_labels)
    test_scores.append(test_score)
    f1_scores.append(f1_score_result)

avg_test_score = np.mean(test_scores)
avg_f1_score = np.mean(f1_score_result)

print('Test: ', avg_test_score)
print('Avg: ', avg_f1_score)
