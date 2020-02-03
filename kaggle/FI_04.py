# Feature seleciton
from sklearn.feature_selection import SelectKBest, f_classif

feature_cols = baseline_data.columns.drop('outcome')

# Keep 5 features
selector = SelectKBest(f_classif, k=5)

feature_cols = baseline_data.columns.drop('outcome')
train, valid, _ = get_data_splits(baseline_data)

## only with train data to avoid data-leakaage
X_new = selector.fit_transform(train[feature_cols], train['outcome'])
X_new

## To obtain dropped features: 
# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(
    selector.inverse_transform(X_new), 
    index=train.index, 
    columns=feature_cols)
selected_features.head()

# This returns a DataFrame with the same index and columns as the training set, 
# but all the dropped columns are filled with zeros.

# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]

# Get the valid dataset with the selected features.
valid_new = valid[selected_columns].head()

# L1 reg: 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

train, valid, _ = get_data_splits(baseline_data)

X, y = train[train.columns.drop("outcome")], train['outcome']

# Set the regularization parameter C=1
logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)
X_new



