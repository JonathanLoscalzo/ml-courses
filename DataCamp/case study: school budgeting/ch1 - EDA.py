# Budget data
df = pd.read_csv('')

# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
# df.FTE.dropna().hist()
plt.hist(df.FTE.dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()

##############
# category pandas. 
# pd.get_dummies(df, prefix_sep="_")
df.dtypes.value_counts()

# Define the lambda function: categorize_label
categorize_label = lambda col: col.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)

###############
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].nunique()

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()

#######

# measure:  
# log loss provides a steep penalty for predictions that are both wrong and confident, 
# Remember, confident and wrong predictions are highly penalized, resulting in a higher log loss.
# The goal of our machine learning models is to minimize this value. 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html
import numpy as np
from numpy import log, clip

def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -log(p)
  else:
    return -log(1 - p)

def compute_log_loss(predicted, true_label):
    from sklearn.metrics import log_loss
    return log_loss(true_label, predicted)

# logloss(0,0.99) => 4.60
# logloss(1,0.99) => 0.01

correct_confident = list([0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05])
correct_not_confident = list([0.65, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35, 0.35, 0.35, 0.35])
wrong_not_confident = list([0.35, 0.35, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.65, 0.65])
wrong_confident = list([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])
actual_labels = list([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])

# Log loss, correct and confident: 0.05129329438755058
# Log loss, correct and not confident: 0.4307829160924542
# Log loss, wrong and not confident: 1.049822124498678
# Log loss, wrong and confident: 2.9957322735539904
# Log loss, actual labels: 9.99200722162646e-15
