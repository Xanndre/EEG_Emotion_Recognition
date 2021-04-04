import pandas as pd


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
