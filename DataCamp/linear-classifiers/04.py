# https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm

## how it looks when only show support vectors

# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X, y)
plot_classifier(X, y, svm, lims=(11,15,0,6))

# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_))
X_small = X[svm.support_]
y_small = y[svm.support_]

# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)
plot_classifier(X_small, y_small, svm_small, lims=(11,15,0,6))

#########################

#transforming features could lead to transform from a non-linearly-separable problem to one which is linearly-separable. 
# When inverse_transform the boundary, what happends? if we use square-function, in the original space could be a ellipsis. 
# Kernel SVM computes feature transformation in a performant computation.
# gamma: smoothness of the boundary. the more greater, the more fit might be

# In the video we saw that increasing the RBF kernel hyperparameter gamma increases training accuracy.
# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X, y)

# Report the best parameters
print("Best CV params", searcher.best_params_)
# Best CV params {'gamma': 0.001}
# Larger values of gamma are better for training accuracy, 
# but cross-validation helped us find something different (and better!).

##############
# C is regularization. 
# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
parameters = {
    'C':[0.1, 1, 10], 
    'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]
}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

# Best CV params {'C': 10, 'gamma': 0.0001}
# Best CV accuracy 0.9988864142538976
# Test accuracy of best grid search hypers: 0.9988876529477196

#######################

# Fundamentals parameters
# linear_model.LogisticRegression
# - C regularization
# - penalty: type of regularization (l1, l2)
# - multi_class: control of type of multi_class
# svm.SVC() (kernel) or svm.LinearSVC()
# - C: inverse regularization
# - kernel: 
# - gamma: inverse controls smoothness

# SGDClassifier: scale well to large datasets
# loss:'log','hinge'
# alpha: 1/C

########################

# We set random_state=0 for reproducibility 
linear_classifier = SGDClassifier(random_state=0)

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
             'loss':['log', 'hinge'], 'penalty':['l1','l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

#############3
# Conclusion: 
# DS is the process of answerring question and making decision based on data.
# The datascience process usually combines several of the following pieces: 
# - data collection
# - data preparation
# - data base design
# - data visualization
# - communication
# - software engineering
# - machine learning