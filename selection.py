from sklearn.feature_selection import SelectKBest, f_classif


def select_band_features(band, data, is_pair):
    band_features = list(filter(lambda x: x.endswith(band), data.columns))
    if not is_pair:
        band_features = list(
            filter(lambda x: not x.startswith('Pair'), band_features))
    return data[band_features].to_numpy()


def select_channel_features(channels_out, data, is_pair):
    channel_features_out = list([col for col in data.columns if any(
        chan in col for chan in channels_out)])

    channel_features = list(set(
        data.columns) - set(channel_features_out) - set(['Valence', 'Arousal', 'Emotion']))

    if not is_pair:
        channel_features = list(
            filter(lambda x: not x.startswith('Pair'), channel_features))

    return data[channel_features].to_numpy()


def select_anova_features(k, x, y):
    return SelectKBest(f_classif, k=k).fit_transform(x, y)
