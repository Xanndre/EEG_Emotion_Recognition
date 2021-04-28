from sklearn.feature_selection import SelectKBest, f_classif, chi2


def select_anova_features(k, x, y):
    return SelectKBest(f_classif, k=k).fit_transform(x, y)


def select_band_features(band, data):
    band_features = list(filter(lambda x: x.endswith(band), data.columns))
    return data[band_features].to_numpy()


def select_channel_features(channels, data):
    channel_features = list(
        filter(lambda x: x.startswith(channels), data.columns))
    return data[channel_features].to_numpy()


def select_chi2_features(k, x, y):
    return SelectKBest(chi2, k=k).fit_transform(x, y)
