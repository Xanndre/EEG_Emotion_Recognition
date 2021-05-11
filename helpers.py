import constants
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc


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

    fig.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.show()


def draw_confusion_matrix(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred.argmax(axis=1)))


def print_precision_recall(y_pred, y_test):
    print('Precision: ', '%.4f' % precision_score(
        y_test, y_pred.argmax(axis=1), average='micro'))
    print('Recall: ', '%.4f' % recall_score(
        y_test, y_pred.argmax(axis=1), average='micro'))


def draw_roc_curve(y_pred, y_test, n_classes):
    y_pred = label_binarize(y_pred.argmax(axis=1), classes=range(n_classes))
    y_test = label_binarize(y_test, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['avg'], tpr['avg'], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc['avg'] = auc(fpr['avg'], tpr['avg'])

    plt.figure()
    plt.plot(fpr['avg'], tpr['avg'],
             label='average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc['avg']),
             color='darkorange', linestyle=':', linewidth=4)

    colors = cycle(['orchid', 'mediumslateblue', 'deepskyblue', 'greenyellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
