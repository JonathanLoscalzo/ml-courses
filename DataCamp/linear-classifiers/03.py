Logistic regression
C: inverse of the regularization logreg
regularization improve test accuracy (sometimes)
making the coefficients smallers

LinearRegresesion
Lasso: L1: performs feature selection ...
Ridge: L2


# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()

# Loop over values of C_value
for C_value in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create LogisticRegression object and fit
    lr = LogisticRegression(C=C_value)
    lr.fit(X_train, y_train)
    
    # Evaluate error rates and append to lists
    train_errs.append( 1.0 - lr.score(X_train, y_train) )
    valid_errs.append( 1.0 - lr.score(X_valid, y_valid) )
    
# Plot results
plt.semilogx(C_values, train_errs, C_values, valid_errs)
plt.legend(("train", "validation"))
plt.show()

# Congrats! As you can see, too much regularization (small C) doesn't work well 
# - due to underfitting - and too little regularization (large C) doesn't work well either 
# - due to overfitting.
#############3 
# Logistic regression and feature selection
# In this exercise we'll perform feature selection on the movie review sentiment data set using L1 regularization.
# Specify L1 regularization
lr = LogisticRegression(penalty='l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))

# Best CV params {'C': 1}
# Total number of features: 2500
# Number of selected features: 1220

##############3
# Identifying the most positive and negative words
# In this exercise we'll try to interpret the coefficients of a logistic regression 
# fit on the movie review sentiment dataset.

# Get the indices of the sorted cofficients
inds_ascending = np.argsort(lr.coef_.flatten()) 
inds_descending = inds_ascending[::-1]

# Print the most positive words
print("Most positive words: ", end="")
for i in range(5):
    print(vocab[inds_descending[i]], end=", ")
print("\n")

# Print most negative words
print("Most negative words: ", end="")
for i in range(5):
    print(vocab[inds_ascending[i]], end=", ")
print("\n")
   
   
    # Most positive words: favorite, superb, noir, knowing, loved, 
    
    # Most negative words: disappointing, waste, worst, boring, lame,
##############3
# Logreg and probabilities.
# <script.py> output: C=1
#     Maximum predicted probability 0.9761229966765974

# <script.py> output: C=0.1
#     Maximum predicted probability 0.8990965659596716
# As you probably noticed, smaller values of C lead to less confident predictions. 
# That's because smaller C means more regularization, which in turn means smaller coefficients, 
# which means raw model outputs closer to zero and, thus, probabilities closer to 0.5 after 
# the raw model output is squashed through the sigmoid function. 


lr = LogisticRegression()
lr.fit(X,y)

# Get predicted probabilities
proba = lr.predict_proba(X)

# Sort the example indices by their maximum probability
proba_inds = np.argsort(np.max(proba,axis=1))

# Show the most confident (least ambiguous) digit
show_digit(proba_inds[-1], lr)

# Show the least confident (most ambiguous) digit
show_digit(proba_inds[0], lr)

############################33

# Multiclass classifier: 
# - one-vs-rest ( it's like binary classification )
# lr0.fit(X, y==0)
# lr1.fit(X, y==1)
# lr2.fit(X, y==2)

# lr0.decision_function(X)[0] # return confidence of the model

# - softmax, cross-entropy, multinomial. 

# multinomial coeff: per feature, per class, summing up intercepts.

# Fit one-vs-rest logistic regression classifier
lr_ovr = LogisticRegression()
lr_ovr.fit(X_train, y_train)

print("OVR training accuracy:", lr_ovr.score(X_train, y_train))
print("OVR test accuracy    :", lr_ovr.score(X_test, y_test))

# Fit softmax classifier
lr_mn = LogisticRegression(multi_class='multinomial', solver = "lbfgs")
lr_mn.fit(X_train, y_train)

print("Softmax training accuracy:", lr_mn.score(X_train, y_train))
print("Softmax test accuracy    :", lr_mn.score(X_test, y_test))

#    OVR training accuracy: 0.9948032665181886
#     OVR test accuracy    : 0.9644444444444444
#     Softmax training accuracy: 1.0
#     Softmax test accuracy    : 0.9688888888888889


##############################333

# Print training accuracies
print("Softmax     training accuracy:", lr_mn.score(X_train, y_train))
print("One-vs-rest training accuracy:", lr_ovr.score(X_train, y_train))

# Create the binary classifier (class 1 vs. rest)
lr_class_1 = LogisticRegression(C=100)
lr_class_1.fit(X_train, y_train==1)

# Plot the binary classifier (class 1 vs. rest)
plot_classifier(X_train, y_train==1, lr_class_1)

    # Softmax     training accuracy: 0.996
    # One-vs-rest training accuracy: 0.916

    # We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
# We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train==1)
plot_classifier(X_train, y_train==1, svm_class_1)

he non-linear SVM works fine with one-vs-rest on this dataset because it learns to "surround" class 1.