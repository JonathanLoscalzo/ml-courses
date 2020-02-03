## ML kaggle, intermediate micro-course

# obtain columns with missings
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as pyplot
cols_with_missing = [
    col for col in X_train.columns if X_train[col].isnull().any()]

# Make new columns indicating what will be imputed (approach loco)
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()


missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# # missing values
# 1) remove columns/rows
# 2) impute by columns
# 3) impute and add column showing if "col" was imputed

# Categorical Variables (encoding)
# ordered and unordered  (ordinal & nominal)

# 2) label encoding
# tree-based models work well with ordinal variables
# sklearn.preprocessing => LabelEncoder

# 1) dummy encoding
# work well with nominal variables
# We set handle_unknown='ignore' to avoid errors when the validation
# data contains classes that aren't represented in the training data,
# and setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if set(
    X_train[col]) == set(X_valid[col])]


# obtain cardinality of categorical columns
{c: len(X_train[c].unique()) for c in object_cols}

# we typically will only one-hot encode columns with relatively low cardinality.

# Pipelines
# - Cleaner Code: Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step.
# - Fewer Bugs: There are fewer opportunities to misapply a step or forget a preprocessing step.
# - Easier to Productionize: It can be surprisingly hard to transition a model from a prototype to something deployable at scale. We won't go into the many related concerns here, but pipelines can help.
# - More Options for Model Validation: You will see an example in the next tutorial, which covers cross-validation.

# cradinality cols with less 10 elements
low_cardinality_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']
                ]


###############

# xgboost plot training evaluation metrics
# retrieve performance metrics
results = my_model_2.evals_result()
epochs = len(results['validation_0']['mae'])
x_axis = range(0, epochs)
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mae'], label='Train')
ax.plot(x_axis, results['validation_1']['mae'], label='Test')
ax.legend()
pylot.xlabel('epochs')
pyplot.ylabel('MAE')
pyplot.title('XGBoost MAE Error')
pyplot.show()

####################################


# Training with GridSearchCV a XGBoost


my_pipeline = Pipeline(
    [
        ('imputer', Imputer()),  # below alternative
        ('xgbrg', XGBRegressor())
    ]
)

param_grid = {
    "xgbrg__n_estimators": [10, 50, 100, 500],
    "xgbrg__learning_rate": [0.1, 0.5, 1],
}

fit_params = {"xgbrg__eval_set": [(val_X, val_y)],
              "xgbrg__early_stopping_rounds": 10,
              "xgbrg__verbose": False}

searchCV = GridSearchCV(
    my_pipeline,
    cv=5,
    param_grid=param_grid,
    fit_params=fit_params
)
searchCV.fit(train_X, train_y)

# Alternative to impute as a preprocessor:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing SimpleImputer, OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
# Pipeline return a transformer piped

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

my_pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),  # below alternative
        ('xgbrg', XGBRegressor())
    ]
)

# for more examples, link: 
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# https://scikit-learn.org/stable/modules/compose.html
# https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html




###
# # Leakage
# Target leakage occurs when your predictors include data that will not be available at the time you make predictions.
# It is important to think about target leakage in terms of the timing or chronological order that data becomes available, 
# not merely whether a feature helps make good predictions.

# Example: took_medicine => got_ill

# To prevent this type of data leakage, any variable updated (or created) 
# after the target value is realized should be excluded.

# Train-Test Contamination
# -> Feature Engineering with whole dataset (and not only with train)
# for example, only impute with the mean calculated from train-dataset

# Exclude de validation data from any type of fitting, including preprocesing. 
# It's easier when we use sklearn pipelines

