
# Restricted and unrestricted decision trees
# For this exercise, we will revisit the Pokémon dataset from the last chapter. Recall that the goal is to predict whether or not a given Pokémon is legendary.

# Here, you will build two separate decision tree classifiers. In the first, you will specify the parameters min_sample_leaf and min_sample_split, but not a maximum depth, so that the tree can fully develop without any restrictions.

# In the second, you will specify some constraints by limiting the depth of the decision tree. By then comparing the two models, you'll better understand the notion of a "weak" learner.

# Build unrestricted decision tree
clf = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9)
clf.fit(X_train, y_train)

# Predict the labels
pred = clf.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)

# Print the F1 score
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))
# F1-Score: 0.583
#============================================

# Build restricted decision tree
clf = DecisionTreeClassifier(max_depth=4,max_features=2, random_state=500)
clf.fit(X_train, y_train)

# Predict the labels
pred = clf.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)

# Print the F1 score
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))
# F1-Score: 0.526

# Correct choice! Model A is a fine-tuned decision tree, with a decent performance on its own. Model B is 'weak', restricted in height and with performance just above 50%.