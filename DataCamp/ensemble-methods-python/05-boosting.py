# Build and fit linear regression model
reg_lm = LinearRegression(normalize=True)
reg_lm.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_lm.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))

# Boosting for predicted revenue
# The initial model got an RMSE of around 7.34. Let's see if we can improve this using an iteration of boosting.

# You'll build another linear regression, but this time the target values are the errors from the base model, calculated as follows:

# y_train_error = pred_train - y_train
# y_test_error = pred_test - y_test
# For this model you'll also use 'popularity' as an additional feature, hoping that it can provide informative patterns than with the 'budget' feature alone. This is available to you as X_train_pop and X_test_pop

# Fit a linear regression model to the previous errors
reg_error = LinearRegression(normalize=True)
reg_error.fit(X_train_pop, y_train_error)

# Calculate the predicted errors on the test set
pred_error = reg_error.predict(X_test_pop)

# Evaluate the updated performance
rmse_error = np.sqrt(mean_squared_error(pred_error, y_test_error))
print('RMSE: {:.3f}'.format(rmse_error))

# ADABOOST
# Instantiate a normalized linear regression model
reg_lm = LinearRegression(normalize=True)

# Build and fit an AdaBoost regressor
reg_ada = AdaBoostRegressor(reg_lm, 12, random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
# RMSE: 7.179

# Build and fit a tree-based AdaBoost regressor (default: decisionTree)
reg_ada = AdaBoostRegressor(n_estimators=12, random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
#RMSE: 5.443

#  Making the most of AdaBoost
# As you have seen, for predicting movie revenue, AdaBoost gives the best results with decision trees as the base estimator.

# In this exercise, you'll specify some parameters to extract even more performance. In particular, you'll use a lower learning rate to have a smoother update of the hyperparameters. Therefore, the number of estimators should increase. Additionally, the following features have been added to the data: 'runtime', 'vote_average', and 'vote_count'.

# Build and fit an AdaBoost regressor
reg_ada = AdaBoostRegressor(
    n_estimators=100, learning_rate=0.01, random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
#RMSE: 5.150

###GradientBoosting
# Build and fit a Gradient Boosting classifier
clf_gbm = GradientBoostingClassifier(100, learning_rate=0.1, random_state=500)
clf_gbm.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = clf_gbm.predict(X_test)

# Evaluate the performance based on the accuracy
acc = accuracy_score(pred, y_test)
print('Accuracy: {:.3f}'.format(acc))

# Get and show the Confusion Matrix
cm = confusion_matrix(pred, y_test)
print(cm)


