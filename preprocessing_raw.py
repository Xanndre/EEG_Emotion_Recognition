import mne
import numpy as np
from xml.dom import minidom
import os


def preprocess(tmin, tmax):
    signals = []
    labels = []
    for dir in os.listdir('Sessions'):
        for file in os.listdir(os.path.join("Sessions/", dir)):
            if 'S_Trial' in file:
                data = mne.io.read_raw_bdf(
                    'Sessions/' + dir + '/' + file, preload=True).crop(tmin, tmax)
                data = data.filter(4, 45., fir_design='firwin',
                                   skip_by_annotation='edge').get_data()[0:32]

                xml_items = minidom.parse(
                    'Sessions/' + dir + '/session.xml').getElementsByTagName('session')
                felt_emo = xml_items[0].attributes['feltEmo'].value
                signals.append(data)
                labels.append(int(felt_emo))
                break

    return np.array(signals), np.array(labels)
