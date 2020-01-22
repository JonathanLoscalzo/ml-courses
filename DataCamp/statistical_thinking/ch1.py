# bee swarm
# Create bee swarm plot with Seaborn's default settings
sns.swarmplot(x='species', y='petal length (cm)', df=df)

# Label the axes
plt.xlabel('species')
plt.ylabel('')

# Show the plot
plt.show()

# ECDF: cumulative distribution
# Define a function with the signature ecdf(data). Within the function definition,
# Compute the number of data points, n, using the len() function.
# The x-values are the sorted data. Use the np.sort() function to perform the sorting.
# The y data of the ECDF go from 1/n to 1 in equally spaced increments. 
# You can construct this using np.arange(). 
# Remember, however, that the end value in np.arange() is not inclusive. 
# Therefore, np.arange() will need to go from 1 to n+1. Be sure to divide this by n.
# The function returns the values x and y.

# ECDF as an alternative to visualization of swarmplots (with a tons of data)
# X valuas are quantity of you are measuring
# Y is the fraction of data points that have a value smaller than the corresponding x-value
# Example: 
# prob vs percent votes of obama. 
# 20 % of the counties had 36% or less vote for obama. 

# ECDF show as percentiles (in this case boxplot are [2.5, 25, 50, 75, 97.5] percentiles )
# ECDF gives a complete picture of how the data are distributed

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
plt.plot(x_vers, y_vers, marker='.', linestyle = "none")

# Label the axes
plt.xlabel('petal length (cm)')
plt.ylabel('ECDF')

# Display the plot
plt.show()

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)

# Plot all ECDFs on the same plot
plt.plot(x_set, y_set, marker='.', linestyle = "none")
plt.plot(x_vers, y_vers, marker='.', linestyle = "none")
plt.plot(x_virg, y_virg, marker='.', linestyle = "none")

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()