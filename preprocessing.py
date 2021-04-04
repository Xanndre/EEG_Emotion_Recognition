import mne
import pandas as pd
from xml.dom import minidom
import os
import helpers
import extraction

TMIN, TMAX = 30, 90
SAMPLING_FREQ = 256
WINDOW = 4 * SAMPLING_FREQ

channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
            'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

pairs = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F7', 'F8'],
         ['F3', 'F4'], ['FC5', 'FC6'], ['FC1', 'FC2'],
         ['T7', 'T8'], ['C3', 'C4'], ['O1', 'O2'], ['CP5', 'CP6'],
         ['CP1', 'CP2'], ['P7', 'P8'], ['P3', 'P4'], ['PO3', 'PO4']]

bands = {'Theta': (4, 8),
         'Slow Alpha': (8, 10),
         'Alpha': (8, 12),
         'Beta': (12, 30),
         'Gamma': (30, 45)}


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

    arousal_values = list(map(helpers.emotion_to_arousal, labels))
    valence_values = list(map(helpers.emotion_to_valence, labels))

    concatenated_df = pd.concat([feature_df, pair_feature_df], axis=1)
    concatenated_df['Emotion'] = labels
    concatenated_df['Arousal'] = arousal_values
    concatenated_df['Valence'] = valence_values
    concatenated_df.to_csv('features.csv', index=False)


if __name__ == "__main__":
    main()
