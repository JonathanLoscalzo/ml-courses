# Condorcet's Jury theorem
# - models are independent
# - each model performs better than random guessing
# - all individual models have similar performance
# (if this rule are correct, it will increase the probabilty and performance of the ensemble).

# Bootstrapping
# - random subsample of dataset
# - using replacement

# Bagging:
# pro's:
# - reduce variance
# - overfitting can be avoided

# con's 
# - computation expensive

# =========================================

# Training with bootstrapping
# Let's now build a "weak" decision tree classifier and train it on a sample of the training set drawn with replacement. This will help you understand what happens on every iteration of a bagging ensemble.

# To take a sample, you'll use pandas' .sample() method, which has a replace parameter. For example, the following line of code samples with replacement from a DataFrame df:

# Take a sample with replacement
X_train_sample = X_train.sample(frac=1.0, replace=True, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# Build a "weak" Decision Tree classifier
clf = DecisionTreeClassifier(max_depth=4, max_features=2, random_state=500)

# Fit the model to the training sample
clf.fit(X_train_sample, y_train_sample)


#==============================

# Build the list of individual models
clf_list = []
for i in range(21):
	clf_list.append(build_decision_tree(X_train, y_train, random_state=i))

# Predict on the test set
pred = predict_voting(clf_list, X_test)

# Print the F1 score
print('F1 score: {:.3f}'.format(f1_score(y_test, pred)))

### BAGGING

# Instantiate the base model
clf_dt = DecisionTreeClassifier(max_depth=4)

# Build and train the Bagging classifier
clf_bag = BaggingClassifier(
  clf_dt,
  21,
  random_state=500)
clf_bag.fit(X_train, y_train)

# Predict the labels of the test set
pred = clf_bag.predict(X_test)

# Show the F1-score
print('F1-Score: {:.3f}'.format(f1_score(y_test, pred)))

# out-of-bag score
# Build and train the bagging classifier
clf_bag = BaggingClassifier(
  clf_dt,
  21,
  oob_score=True,
  random_state=500)
clf_bag.fit(X_train, y_train)

# Print the out-of-bag score
print('OOB-Score: {:.3f}'.format(clf_bag.oob_score_))

# Evaluate the performance on the test set to compare
pred = clf_bag.predict(X_test)
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, pred)))

#    OOB-Score: 0.933
#     Accuracy: 0.963

#=======================================
# Build a balanced logistic regression
clf_lr = LogisticRegression(class_weight='balanced')

# Build and fit a bagging classifier
clf_bag = BaggingClassifier(clf_lr, oob_score=True, max_features=10, random_state=500)
clf_bag.fit(X_train, y_train)

# Evaluate the accuracy on the test set and show the out-of-bag score
pred = clf_bag.predict(X_test)
print('Accuracy:  {:.2f}'.format(accuracy_score(y_test, pred)))
print('OOB-Score: {:.2f}'.format(clf_bag.oob_score_))

# Print the confusion matrix
print(confusion_matrix(y_test, pred))
    # Accuracy:  0.73
    # OOB-Score: 0.65

#================
# Build a balanced logistic regression
clf_base = LogisticRegression(class_weight='balanced', random_state=42)

# Build and fit a bagging classifier with custom parameters
clf_bag = BaggingClassifier(clf_base, 500, max_features=10, max_samples=.65, bootstrap=False, random_state=500)
clf_bag.fit(X_train, y_train)

# Calculate predictions and evaluate the accuracy on the test set
y_pred = clf_bag.predict(X_test)
print('Accuracy:  {:.2f}'.format(accuracy_score(y_test, y_pred)))

# Print the classification report
print(classification_report(y_test, y_pred))