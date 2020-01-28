# how good is your model?

Imbalanced Datasets
Accuracy is not a good metric in these cases.
- fails at its original purpose
Need more nuanced metrics

confusion matrix 
    0   1   predicted
0   TN  FP
1   FN  TP
real or actual

precision: tp / tp + fp
recall: tp / tp + fn
f1: 2 * precision * recall / (precision + recall)

############
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# By analyzing the confusion matrix and classification report, 
# you can get a much better understanding of your 
# classifier's performance.

####
Logistic regression (classification problems)
"logreg" outputs probabilities
if p is greater than 0.5 labeled true/1, 
if p is less than 0.5 labeled false/0

- logreg produces a linear decision boundary
- by default, logreg threshold is 0.5
- varying the threshold, we could visualize ROC curve

logreg.predict_proba(data)[:,1] => probability of predicting labels being 1.

# the Train-Test-Split/Instantiate/Fit/Predict paradigm applies to all classifiers and regressors - which are known in scikit-learn as 'estimators'.

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Logistic regression is used in a variety of machine learning applications and will become a vital part of your data science toolbox.

############## 
# while ROC curves provide a way to visually evaluate models.
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

########

#Study the precision-recall curve and then consider the statements given below.  

- A recall of 1 corresponds to a classifier with a low threshold in which all females who contract diabetes were correctly classified as such, at the expense of many misclassifications of those who did not have diabetes.
- Precision is undefined for a classifier which makes no positive predictions, that is, classifies everyone as not having diabetes.
- When the threshold is very close to 1, precision is also 1, because the classifier is absolutely certain about its predictions.

############
# Area under ROC curve (AUC)
# larger area under ROC curve, the better our model is
# from sklearn.metrics import roc_auc_score

# AUC using cross-validation
# from sklearn.model_selection import cross_val_score
# cross_val_score(est, X,y, cv=5, scoring='roc_auc')
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
#AUC: 0.8254806777079764
#AUC scores computed using 5-fold cross-validation: [0.80148148 0.8062963  0.81481481 0.86245283 0.8554717 ]

###############################
#  
# Hyperparameter tuning
# Linear: parameters (learned by the model)
# ridge/lasso: choosing alpha
# knn: choosing neighbors

# alpha and k are hyperparameters, them cannot be fitted by the model. 

# IT IS essential to use cross-validation, as using
# train_test_split alone would risk overfitting the hyperparameter to the test set.add

# Grid-Search cross-validation: GridSearchCV
# from sklearn.model_selection import GridSearchCV

# model_cv.best_params_
# model_cv.best_score_


# Like the alpha parameter of lasso and ridge regularization 
# that you saw earlier, logistic regression also has a regularization
#  parameter: C. 
#  C controls the inverse of the regularization strength, 
#  and this is what you will tune in this exercise. 
#  A large C can lead to an overfit model, 
#  while a small C can lead to an underfit model.
# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = { 'C': c_space }

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))
# Tuned Logistic Regression Parameters: {'C': 3.727593720314938}
# Best score is 0.7708333333333334

# Hyperparameter tuning with RandomizedSearchCV
# GridSearchCV can be computationally expensive, especially if you are searching 
# over a large hyperparameter space and dealing with multiple hyperparameters. 
# A solution to this is to use RandomizedSearchCV, in which not all hyperparameter 
# values are tried out.

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
#Tuned Decision Tree Parameters: {'criterion': 'gini', 'max_depth': 3, 'max_features': 5, 'min_samples_leaf': 2}
#Best score is 0.7395833333333334

# RandomizedSearchCV will never outperform GridSearchCV. Instead, it is valuable because it saves on computation time.

####################33
# First: Hold-out evaluation data
# how well can the model perform on never seen data?
# Using ALL data for cv is not ideal
# split data into training and hold-out (or evaluation) set at the beginning
# perform grid-search on cross-validation on training
# choose best hyperparameters and evaluate on hold-out set

## Hold-out set Classification
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state =42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

#Tuned Logistic Regression Parameter: {'C': 0.4393970560760795, 'penalty': 'l1'}
#Tuned Logistic Regression Accuracy: 0.7652173913043478

## Hold-out set Regression
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state = 42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

