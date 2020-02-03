### First: baseline model
- encode categorical features (onehot or label)
- 


## Categorical encoding: others
- count encoding: 
- target encoding: 
- SVD: singular value decomposition

# see. FI_02.py

### Count enconding
Count encoding replaces each categorical value with the number of times it appears in the dataset. 
For example, if the value "GB" occured 10 times in the country feature,
then each "GB" would be replaced with the number 10.

see: https://contrib.scikit-learn.org/categorical-encoding/


### Target encoding
Target encoding replaces a categorical value with the average value of the target 
for that value of the feature.

This technique uses the targets to create new features. So including the validation 
or test data in the target encodings would be a form of target leakage.
# https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html

### CatBoost Encoding
This is similar to target encoding in that it's based on the target probablity for a given value. 
However with CatBoost, for each row, the target probability is calculated only from the rows before it.


## 03 - Feature Generation
### Interactions
Combine categorical features: CA, Music => CA_MUSIC
It combination can provide correlation information between cat variables (after encode)

### Numbers rolling windows
rolling windows of 7-days before (such as projects), to add information of time-series.

add information since and event ocurred. (such as categories promoted)

### Numerical Features: 
# Common choices for this are the square root and natural logarithm. 
# These transformations can also help constrain outliers.

# The log transformation won't help our model since tree-based models are scale invariant. 
# However, this should help if we had a linear model or neural network.

## 04 - feature selection
# More features will be able to lead overfitting.
# The more features you have, the longer it will take your model to train. 

# Univariate Feature Selection
# The simplest and fastest methods are based on univariate statistical tests. For each feature, measure how strongly the target depends on the feature using a statistical test like  χ2  or ANOVA.

# From the scikit-learn feature selection module, feature_selection.SelectKBest returns the K best features given some scoring function.
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

# The statistical tests are calculated using all of the data. 
# This means information from the validation and test sets could influence the features we keep, 
# introducing a source of leakage

# L1 regularization
# Univariate methods consider only one feature at a time when making a selection decision.
# Instead, we can make our selection using all of the features 
# by including them in a linear model with L1 regularization. 
# L1: lasso (coeff penalty size)
# L2: Ridge (square of the coeff penalty size)

# For regression problems you can use sklearn.linear_model.Lasso, or sklearn.linear_model.LogisticRegression for classification. 
# These can be used along with sklearn.feature_selection.SelectFromModel to select the non-zero coefficients. Otherwise, the code is similar to the univariate tests.





# You could use something like RandomForestClassifier or ExtraTreesClassifier to find feature importances. 
# SelectFromModel can use the feature importances to find the best features.
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html


#  To select a certain number of features with L1 regularization, you need to find the regularization parameter that leaves the desired number of features. To do this you can iterate over models with different regularization parameters from low to high and choose the one that leaves K features. 
#  Note that for the scikit-learn models C is the inverse of the regularization strength.