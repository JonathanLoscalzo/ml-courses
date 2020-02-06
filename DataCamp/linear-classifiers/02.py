### coefficients

dot product 
x@y => x dot y => sum(x * y)

coeff * features  + intercept = raw model output

model checks if it is positivo (true) or negative (false)
raw model output determines which side of the boundary is located some point.


# # loss function
# least squarse
# sum ( true target value - predicted target value)²

# 0-1 loss: number of errors of the model (precision), hard to minimize it.iterable

# Minimizing a loss: 
# from scipy.optimize import minimize

# minimize(np.square, 2).x

# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true - y_i_pred)**2
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)

#############################
#https://algorithmia.com/blog/introduction-to-loss-functions
Loss Function diagrams
squared error: for linear regression, "prefer" the true label...
Logloss: for classifiers, penalize all.

# Mathematical functions for logistic and hinge losses
def log_loss(raw_model_output):
   return np.log(1+np.exp(-raw_model_output))
def hinge_loss(raw_model_output):
   return np.maximum(0,1-raw_model_output)

# Create a grid of values and plot
grid = np.linspace(-2,2,1000)
plt.plot(grid, log_loss(grid), label='logistic')
plt.plot(grid, hinge_loss(grid), label='hinge')
plt.legend()
plt.show()
############33

def log_loss(raw_model_output):
   return np.log(1+np.exp(-raw_model_output))

# The logistic loss, summed over training examples
def my_loss(w):
    s = 0
    for i in range(len(X)):
        raw_model_output = w@X[i]
        s = s + log_loss(raw_model_output * y[i])
    return s

# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(fit_intercept=False, C=1000000).fit(X,y)
print(lr.coef_)