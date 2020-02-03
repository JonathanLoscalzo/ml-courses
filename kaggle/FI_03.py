# Feature Generation

def count_past_events(series):
    return pd.Series(range(series.size), index=series).rolling('6h').count() - 1

def time_diff(series):
    """ Returns a series with the time since the last timestamp in seconds 
    Example of use: clicks.groupby('ip')['click_time'].transform(time_diff)
    """
    return series.diff().dt.total_seconds()

def previous_attributions(series):
    """ Returns a series with the """
    #print(series.expanding(min_periods=2).sum() - series)
    return series.expanding(min_periods=2).sum() - series

#####################################

import itertools
cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)

# Iterate through each pair of features, combine them into interaction features
for col1, col2 in itertools.combinations(cat_features, 2):
    new_col_name = f'{col1}_{col2}'
    interactions[new_col_name] = clicks[col1].map(str).str.cat(clicks[col2].map(str), sep='_')
    encoder = preprocessing.LabelEncoder()
    interactions[new_col_name] = encoder.fit_transform(interactions[new_col_name])

