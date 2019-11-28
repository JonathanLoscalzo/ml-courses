# Instantiate a Naive Bayes classifier
clf_nb = GaussianNB()

# Fit the model to the training set
clf_nb.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = clf_nb.predict(X_test)

# Evaluate the performance using the accuracy score
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred)))
# acc: 0.97

# ===============================
# Instantiate a 5-nearest neighbors classifier with 'ball_tree' algorithm
clf_knn = KNeighborsClassifier(5, algorithm='ball_tree')

# Fit the model to the training set
clf_knn.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = clf_knn.predict(X_test)

# Evaluate the performance using the accuracy score
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred)))
#acc: 1

# ===============================
# Build and fit a Decision Tree classifier
clf_dt = DecisionTreeClassifier(min_samples_leaf = 3, min_samples_split = 9, random_state=500)
clf_dt.fit(X_train, y_train)

# Build and fit a 5-nearest neighbors classifier using the 'Ball-Tree' algorithm
clf_knn = KNeighborsClassifier(5, algorithm='ball_tree')
clf_knn.fit(X_train, y_train)

# Evaluate the performance using the accuracy score
print('Decision Tree: {:0.4f}'.format(accuracy_score(y_test, clf_dt.predict(X_test))))
print('5-Nearest Neighbors: {:0.4f}'.format(accuracy_score(y_test, clf_knn.predict(X_test))))
# Decision Tree: 0.5930
# 5-Nearest Neighbors: 0.5679

# ===============================
# Create a Pandas DataFrame with the predictions
pred_df = pd.DataFrame({
	'pred_dt':pred_dt,
    'pred_knn': pred_knn
}, index=X_train.index)

# Concatenate X_train with the predictions DataFrame
X_train_2nd = pd.concat([X_train, pred_df], axis=1)

# Build the second-layer meta estimator
clf_stack = DecisionTreeClassifier(random_state=500)
clf_stack.fit(X_train_2nd,y_train)

# Create a Pandas DataFrame with the predictions (from X_test)
pred_df = pd.DataFrame({
	'pred_dt':pred_dt,
    'pred_knn': pred_knn
}, index=X_test.index)

# Concatenate X_test with the predictions DataFrame
X_test_2nd = pd.concat([X_test, pred_df], axis=1)

# Obtain the final predictions from the second-layer estimator
pred_stack = clf_stack.predict(X_test_2nd)

# Evaluate the new performance on the test set
print('Accuracy: {:0.4f}'.format(accuracy_score(y_test, pred_stack)))
#Accuracy: 0.5993