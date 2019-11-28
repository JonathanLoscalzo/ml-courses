# http://rasbt.github.io/mlxtend/

# Instantiate the first-layer classifiers
clf_dt = DecisionTreeClassifier(min_samples_leaf = 3, min_samples_split = 9, random_state=500)
clf_knn = KNeighborsClassifier(5, algorithm="ball_tree")

# Instantiate the second-layer meta classifier
clf_meta = DecisionTreeClassifier(random_state=500)

# Build the Stacking classifier
clf_stack = StackingClassifier(
    classifiers=[clf_dt, clf_knn],
    meta_classifier=clf_meta,
    use_features_in_secondary=True)
clf_stack.fit(X_train, y_train)

# Evaluate the performance of the Stacking classifier
pred_stack = clf_stack.predict(X_test)
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred_stack)))
# Accuracy: 0.5993

#=======================000

# Instantiate the 1st-layer regressors
reg_dt = DecisionTreeRegressor(min_samples_leaf = 11, min_samples_split = 33, random_state=500)
reg_lr = LinearRegression(normalize=True)
reg_ridge = Ridge(random_state=500)

# Instantiate the 2nd-layer regressor
reg_meta = LinearRegression()

# Build the Stacking regressor
reg_stack = StackingRegressor(
    meta_regressor=reg_meta,
    regressors=[reg_dt, reg_lr, reg_ridge],
    use_features_in_secondary=True)
reg_stack.fit(X_train, y_train)

# Evaluate the performance on the test set using the MAE metric
pred = reg_stack.predict(X_test)
print('MAE: {:.3f}'.format(mean_absolute_error(y_test, pred)))
#MAE: 0.587


#=======================000
# Create the first-layer models
clf_knn = KNeighborsClassifier(5, algorithm='ball_tree')
clf_dt = DecisionTreeClassifier(min_samples_leaf = 5, min_samples_split = 15, random_state=500)
clf_nb = GaussianNB()

# Create the second-layer model (meta-model)
clf_lr = LogisticRegression()

# Create and fit the stacked model
clf_stack = StackingClassifier(
    classifiers=[clf_knn, clf_dt, clf_nb],
    meta_classifier=clf_lr,
    use_features_in_secondary=True)
clf_stack.fit(X_train, y_train)

# Evaluate the stacked model’s performance
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, clf_stack.predict(X_test))))