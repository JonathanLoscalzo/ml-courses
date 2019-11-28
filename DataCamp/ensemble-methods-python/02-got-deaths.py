# Predicting GoT deaths
# While the target variable does not have any missing values, other features do. As the focus of the course is not on data cleaning and preprocessing, we have already done the following preprocessing for you:

# Replaced NA values with 0.
# Replace negative values of age with 0.
# Replace NA values of age with the mean.
# Let's now build an ensemble model using the averaging technique. The following individual models have been built:

# Logistic Regression (clf_lr).
# Decision Tree (clf_dt).
# Support Vector Machine (clf_svm).
# As the target is binary, all these models might have good individual performance. Your objective is to combine them using averaging. Recall from the video that this is the same as a soft voting approach, so you should still use the VotingClassifier().

# Build the individual models
clf_lr = LogisticRegression(class_weight='balanced')
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf_svm = SVC(probability=True, class_weight='balanced', random_state=500)

# List of (string, estimator) tuples
estimators = [('lr', clf_lr),('dt', clf_dt), ('svm', clf_svm)]

# Build and fit an averaging classifier
clf_avg = VotingClassifier(estimators, voting='soft')
clf_avg.fit(X_train, y_train)

# Evaluate model performance
acc_avg = accuracy_score(y_test,  clf_avg.predict(X_test))
print('Accuracy: {:.2f}'.format(acc_avg))


# =======================================================================
# Soft vs. hard voting
# You've now practiced building two types of ensemble methods: Voting and Averaging (soft voting). Which one is better? It's best to try both of them and then compare their performance. Let's try this now using the Game of Thrones dataset.

# Three individual classifiers have been instantiated for you:

# A DecisionTreeClassifier (clf_dt).
# A LogisticRegression (clf_lr).
# A KNeighborsClassifier (clf_knn).
# Your task is to try both voting and averaging to determine which is better.

# List of (string, estimator) tuples
estimators = [('dt', clf_dt), ('lr', clf_lr), ('knn', clf_knn)]

# Build and fit a voting classifier
clf_vote = VotingClassifier(estimators, voting='hard')
clf_vote.fit(X_train, y_train)

# Build and fit an averaging classifier
clf_avg = VotingClassifier(estimators, voting='soft')
clf_avg.fit(X_train, y_train)

# Evaluate the performance of both models
acc_vote = accuracy_score(y_test, clf_vote.predict(X_test))
acc_avg = accuracy_score(y_test,  clf_avg.predict(X_test))
print('Voting: {:.2f}, Averaging: {:.2f}'.format(acc_vote, acc_avg))