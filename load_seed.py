from constants import BANDS, CHANNELS_SEED, SEED_SESSION1_LABEL, SEED_SESSION2_LABEL, SEED_SESSION3_LABEL
from sklearn import preprocessing
import helpers
import scipy.io
import os
import numpy as np
import pandas as pd

N_FEATURES = len(BANDS) * len(CHANNELS_SEED)
FEATURES_TYPE = 'de_movingAve'
FILE_NAME = 'seed_features_de_movingAve.csv'


def assign_labels(dir):
    if dir == '1':
        return SEED_SESSION1_LABEL
    elif dir == '2':
        return SEED_SESSION2_LABEL
    else:
        return SEED_SESSION3_LABEL


def main():
    merged_data = np.array([]).reshape(0, N_FEATURES)
    merged_labels = []

    for dir in os.listdir('../data/eeg_feature_smooth'):
        labels = assign_labels(dir)

        for file in os.listdir(os.path.join('../data/eeg_feature_smooth/', dir)):

            data = scipy.io.loadmat(os.path.join(
                '../data/eeg_feature_smooth/', dir, file))

            data_keys = list(
                filter(lambda x: x.startswith(FEATURES_TYPE), data.keys()))

            counter = 0
            for key, value in data.items():
                if key in data_keys:
                    reshaped_features = value.transpose(
                        1, 0, 2).reshape(value.shape[1], -1)

                    for _ in range(reshaped_features.shape[0]):
                        merged_labels.append(labels[counter])
                    counter += 1
                    merged_data = np.vstack([merged_data, reshaped_features])

    df = pd.DataFrame(
        merged_data, columns=helpers.get_column_names(CHANNELS_SEED, BANDS))

    data_normalized = preprocessing.MinMaxScaler().fit_transform(df.values)
    df = pd.DataFrame(data_normalized, columns=df.columns)

    df['Label'] = merged_labels
    df.to_csv(FILE_NAME, index=False)


if __name__ == "__main__":
    main()
