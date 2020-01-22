# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)

# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')

# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

# Print the result
print(ptiles_vers)

# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

# Show the plot
plt.show()


# Create box plot with Seaborn's default settings
sns.boxplot(x='species', y='petal length (cm)', data=df)

# Label the axes
plt.xlabel('species')
plt.ylabel('petal length (cm)')

# Show the plot
plt.show()

# _________________________________________________________________________
#Variance: is the average of the square distance of the mean
# 1/n * sum [ ( xi - x_mean)²]
# or mean [ (x1 - x_mean)² ] 

# Standard Deviation: sqrt ( variance ( data ) ) 
# it has the same unit as the data.

# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)

# Square the differences: diff_sq
diff_sq = differences ** 2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Print the results
print(variance_explicit, variance_np)

# Compute the variance: variance
variance = np.var(versicolor_petal_length)

# Print the square root of the variance
print(np.sqrt(variance))

# Print the standard deviation
print(np.std(versicolor_petal_length))

# __________________________________________________________________

# Make a scatter plot
plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')

# Label the axes
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')


# Show the result
plt.show()


#_____________________________________________________________________
#CH2_slides
#Covariance: 
# 1/n * sum ( x - x_mean )* (y - y_mean)
# Positive correlated: When X is high, so is Y

# # If we want a measure of how two variables depend on each other, 
# we want it to be dimensionless (that is to not have units)
# Pearson: (rho greek)
# cov / (std(x) * std(y))

# "variability due to codependence independent variable"
# from -1 to 1
# 0 no correlation

# Look at the spread in the x-direction in the plots: The plot with the largest spread is the one that has the highest variance.
# High covariance means that when x is high, y is also high, and when x is low, y is also low.
# Negative covariance means that when x is high, y is low, and when x is low, y is high.

# - Variance related to disperse data?  
# - Covariance related to correlated data ? 

# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

# Print covariance matrix
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1]

# Print the length/width covariance
print(petal_cov)

#_________________________________________________________________

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat

    corr_mat = np.corrcoef(x,y)
    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r

r = pearson_r(versicolor_petal_length,versicolor_petal_width)
# Print the result
print(r)
