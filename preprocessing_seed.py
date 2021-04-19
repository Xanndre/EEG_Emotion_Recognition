from constants import BANDS_SEED, CHANNELS_SEED
from sklearn import preprocessing
import scipy.io
import os
import numpy as np
import pandas as pd

channel_names = CHANNELS_SEED
band_names = BANDS_SEED

session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1,
                  2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3,
                  2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2,
                  1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]

col_names = []
for chan_name in channel_names:
    for band_name in band_names:
        col_name = chan_name + '_' + band_name
        col_names.append(col_name)

all_val = np.array([]).reshape(0, 310)
shape_values = []
all_labels = []

for dir in os.listdir('../data/eeg_feature_smooth'):
    if dir == '1':
        labels = session1_label
    elif dir == '2':
        labels = session2_label
    else:
        labels = session3_label

    for file in os.listdir(os.path.join('../data/eeg_feature_smooth/', dir)):
        data = scipy.io.loadmat(os.path.join(
            '../data/eeg_feature_smooth/', dir, file))
        data_keys = list(
            filter(lambda x: x.startswith('psd_LDS'), data.keys()))
        counter = 0
        for key, value in data.items():
            if key in data_keys:
                reshaped_val = value.transpose(
                    1, 0, 2).reshape(value.shape[1], -1)
                shape_values.append(reshaped_val.shape[0])
                for i in range(reshaped_val.shape[0]):
                    all_labels.append(labels[counter])
                counter += 1
                all_val = np.vstack([all_val, reshaped_val])


df = pd.DataFrame(all_val, columns=col_names)
x_scaled = preprocessing.MinMaxScaler().fit_transform(df.values)
df = pd.DataFrame(x_scaled, columns=df.columns)
df['Label'] = all_labels
df.to_csv('seed_features.csv', index=False)
