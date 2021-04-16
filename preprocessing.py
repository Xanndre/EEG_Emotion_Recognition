import mne
import pandas as pd
from xml.dom import minidom
import os
import helpers
import constants
import extraction

TMIN, TMAX = 30, 90
SAMPLING_FREQ = 256
WINDOW = 4 * SAMPLING_FREQ

channels = constants.CHANNELS
pairs = constants.PAIRS
bands = constants.BANDS


def main():
    labels = []
    feature_df = helpers.create_empty_feature_df(channels, bands)
    pair_feature_df = helpers.create_empty_pair_feature_df(pairs, bands)

    for dir in os.listdir('Sessions'):
        for file in os.listdir(os.path.join("Sessions/", dir)):
            if 'S_Trial' in file:
                data = mne.io.read_raw_bdf(
                    'Sessions/' + dir + '/' + file, preload=True).crop(TMIN, TMAX)
                data = data.filter(4, 45., fir_design='firwin',
                                   skip_by_annotation='edge').get_data()[0:32]

                band_power_values = extraction.get_band_power_values(
                    data, bands, SAMPLING_FREQ, WINDOW)
                feature_df.loc[len(feature_df.index)] = band_power_values

                pair_feature_df.loc[len(pair_feature_df.index)] = extraction.get_band_power_differences(
                    pairs, bands, pd.Series(band_power_values, index=feature_df.columns))

                xml_items = minidom.parse(
                    'Sessions/' + dir + '/session.xml').getElementsByTagName('session')
                felt_emo = xml_items[0].attributes['feltEmo'].value
                labels.append(int(felt_emo))

                break

    concatenated_df = pd.concat([feature_df, pair_feature_df], axis=1)

    concatenated_df['Emotion'] = labels
    concatenated_df['Arousal'] = list(map(helpers.emotion_to_arousal, labels))
    concatenated_df['Valence'] = list(map(helpers.emotion_to_valence, labels))

    concatenated_df.to_csv('features.csv', index=False)


if __name__ == "__main__":
    main()
