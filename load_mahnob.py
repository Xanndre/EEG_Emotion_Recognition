from constants import BANDS, CHANNELS
from sklearn import preprocessing
from xml.dom import minidom
import pandas as pd
import numpy as np
import helpers
import extraction
import mne
import os


N_WINDOWS = 30
SAMPLING_FREQ = 200
# TMIN, TMAX = 30, 90
FILE_NAME = 'mahnob_features.csv'
LABEL_COLUMNS = ['Emotion', 'Arousal', 'Valence',
                 'Arousal_Mapped', 'Valence_Mapped']

col_names = helpers.get_column_names(CHANNELS, BANDS)


def main():
    merged_df = pd.DataFrame(columns=col_names)

    for dir in os.listdir('../data/Sessions'):
        for file in os.listdir(os.path.join("../data/Sessions/", dir)):
            if 'S_Trial' in file:

                data = mne.io.read_raw_bdf(
                    '../data/Sessions/' + dir + '/' + file, preload=True)
                # .crop(TMIN, TMAX)

                data = data.resample(200, npad="auto").filter(
                    1, 75., fir_design='firwin', skip_by_annotation='edge').get_data()[0:32]

                df = extraction.extract_features(
                    data, N_WINDOWS, SAMPLING_FREQ, BANDS, col_names)

                data_normalized = preprocessing.MinMaxScaler().fit_transform(df.values)
                df = pd.DataFrame(data_normalized, columns=df.columns)

                xml_items = minidom.parse(
                    '../data/Sessions/' + dir + '/session.xml').getElementsByTagName('session')

                felt_emo = xml_items[0].attributes['feltEmo'].value
                felt_emo_array = np.full(
                    shape=N_WINDOWS, fill_value=felt_emo, dtype=np.int)
                felt_arousal = xml_items[0].attributes['feltArsl'].value
                felt_valence = xml_items[0].attributes['feltVlnc'].value

                df['Arousal'] = np.full(
                    shape=N_WINDOWS, fill_value=int(felt_arousal)-1, dtype=np.int)
                df['Valence'] = np.full(
                    shape=N_WINDOWS, fill_value=int(felt_valence)-1, dtype=np.int)
                df['Arousal_Mapped'] = list(
                    map(helpers.emotion_to_arousal, felt_emo_array))
                df['Valence_Mapped'] = list(
                    map(helpers.emotion_to_valence, felt_emo_array))
                df['Emotion'] = list(
                    map(helpers.emotion_to_emotion, felt_emo_array))
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                break

    merged_df[LABEL_COLUMNS] = merged_df[LABEL_COLUMNS].astype(int)
    merged_df.to_csv(FILE_NAME, index=False)


if __name__ == "__main__":
    main()
