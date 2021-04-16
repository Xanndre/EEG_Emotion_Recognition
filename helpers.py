import pandas as pd
import matplotlib.pyplot as plt


def create_empty_feature_df(channels, bands):
    col_names = []
    for channel in channels:
        for band in bands:
            col_name = channel + '_' + band
            col_names.append(col_name)
    return pd.DataFrame(columns=col_names)


def create_empty_pair_feature_df(pairs, bands):
    col_names = []
    for pair in pairs:
        for band in bands:
            if band != 'Slow Alpha':
                col_name = pair[0] + '_' + pair[1] + '_' + band
                col_names.append(col_name)
    return pd.DataFrame(columns=col_names)


def emotion_to_arousal(emotion):
    if emotion in [0, 2, 5]:
        return 0
    elif emotion in [4, 11]:
        return 1
    else:
        return 2


def emotion_to_valence(emotion):
    if emotion in [1, 2, 3, 5, 12]:
        return 0
    elif emotion in [0, 6]:
        return 1
    else:
        return 2


def draw_plot(history):
    _, axes = plt.subplots(nrows=2, ncols=2)

    axes[0, 0].plot(history.history['accuracy'])
    axes[0, 0].set_title('accuracy')

    axes[0, 1].plot(history.history['loss'])
    axes[0, 1].set_title('loss')

    axes[1, 0].plot(history.history['val_loss'])
    axes[1, 0].set_title('val_loss')

    axes[1, 1].plot(history.history['val_accuracy'])
    axes[1, 1].set_title('val_accuracy')

    plt.show()
