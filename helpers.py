import constants
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score


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


def emotion_to_emotion(emotion):
    if emotion == 11:
        return 7
    elif emotion == 12:
        return 8
    else:
        return emotion


def get_column_names(channels, bands):
    col_names = []
    for channel in channels:
        for band in bands:
            col_name = channel + '_' + band
            col_names.append(col_name)
    return col_names


def get_channels(features_type):
    if features_type == 'channels_4':
        return constants.CHANNELS_4
    elif features_type == 'channels_6':
        return constants.CHANNELS_6
    elif features_type == 'channels_7':
        return constants.CHANNELS_7
    elif features_type == 'channels_8':
        return constants.CHANNELS_8
    elif features_type == 'channels_9':
        return constants.CHANNELS_9
    elif features_type == 'channels_10':
        return constants.CHANNELS_10
    elif features_type == 'channels_11':
        return constants.CHANNELS_11
    elif features_type == 'channels_28':
        return constants.CHANNELS_28
    elif features_type == 'channels_42':
        return constants.CHANNELS_42
    elif features_type == 'channels_37':
        return constants.CHANNELS_37
    elif features_type == 'channels_50':
        return constants.CHANNELS_50
    else:
        return constants.CHANNELS_32


def draw_plot(history):
    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].plot(history.history['accuracy'])
    axes[0].set_title('train accuracy')

    axes[1].plot(history.history['loss'])
    axes[1].set_title('train loss')

    # axes[1, 0].plot(history.history['val_loss'])
    # axes[1, 0].set_title('validation loss')

    # axes[1, 1].plot(history.history['val_accuracy'])
    # axes[1, 1].set_title('validation accuracy')

    fig.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.show()


def draw_confusion_matrix(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred.argmax(axis=1)))


def print_precision_recall(y_pred, y_test):
    print('Precision: ', '%.4f' % precision_score(
        y_test, y_pred.argmax(axis=1), average='micro'))
    print('Recall: ', '%.4f' % recall_score(
        y_test, y_pred.argmax(axis=1), average='micro'))
