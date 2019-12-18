import os
import pandas as pd
import matplotlib.pyplot as plt

path = "data/"
file_list = os.listdir(path)
col = ['accx','accy','accz','lx','ly','lz']

for file_name in file_list:
    file_path = os.path.join(path, file_name)
    data = pd.read_csv(file_path, names=col, delimiter=',').values

    accx = data[:,0]
    accy = data[:,1]
    accz = data[:,2]
    lx = data[:,3]
    ly = data[:,4]
    lz = data[:,5]

    figure, axes = plt.subplots(6, sharex = True)
    data_name = ['Acc X', 'Acc Y', 'Acc Z', 'Linear X', 'Linear Y', 'Linear Z']

    for idx, ax in enumerate(axes):
        ax.plot(data[1000:1500, idx])
        ax.set_title(data_name[idx])
        ax.set_yticks([int(data[1000:1500, idx].min()), int(data[1000:1500, idx].max())])
        ax.axvline(100, color = 'k', linestyle = 'dotted', label = 'Period')
        ax.axvline(200, color = 'k', linestyle = 'dotted')
        ax.axvline(300, color = 'k', linestyle = 'dotted')
        ax.axvline(400, color = 'k', linestyle = 'dotted')

    plt.legend(loc = 1)
    plt.tight_layout()
    plt.show()
