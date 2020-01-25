# Hypothesis testing
# Assessment of how reasonable the observed data are assuming a hypothesis is true
# Hypothesis is Null Hypothesis

#Permutation
# Random reordering of entries in a array
# permutable sample

# np.random.permutation()

# the null hypothesis is that the distributions for the two groups are the same


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate([data1, data2])

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

# visualizing permutaiton sampling
for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_june,rain_november)

    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_june)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

# We expect these might be differently distributed, 
# so we will take permutation samples to see how their ECDFs would look if they were identically distributed.

# Notice that the permutation samples ECDFs overlap and give a purple haze. 
# None of the ECDFs from the permutation samples overlap with the observed data, 
# suggesting that the hypothesis is not commensurate with the data. 
# June and November rainfall are not identically distributed.

####################################################################
## Test statistics and p-values
# Hypothesis Testing: assessment of how reasonable the observed data are assuming a hypothesis is true

# Test statistic: a single number that can be computed from observed data and from data you simulate under the null hypothesis
# It serves to compare between two (observed data & hypothesis)

# p-value: the probability of getting a value of your test statistics 
# that is at least as extreme as what was observed, under the assumption the null hypothesis is true
# - is the probability of seeing the apparent effect if the null hypothesis is true.
# - NOT the probability that the null hypothesis is true

# significal statistical: low-pvalues. 
# which means that it is unlikely to have occurred by chance. (poco probable que haya ocurrido por casualidad)

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

# Make bee swarm plot
_ = sns.swarmplot(x="ID", y="impact_force", data = df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()

# Permutation test on frog data
# The average strike force of Frog A was 0.71 Newtons (N), and that of Frog B was 0.42 N for a difference of 0.29 N. 
# It is possible the frogs strike with the same force and this observed difference was by chance. 
# You will compute the probability of getting at least a 0.29 N difference in mean 
# strike force under the hypothesis that the distributions of strike forces for the two frogs are identical. 7
# We use a permutation test with a test statistic of the difference of means to test this hypothesis.

# For your convenience, the data has been stored in the arrays force_a and force_b

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)
# The p-value tells you that there is about a 0.6% chance 
# that you would get the difference of means observed in the experiment 
# if frogs were exactly the same. 
# A p-value below 0.01 is typically said to be "statistically significant," but: warning! warning! warning! 
# You have computed a p-value; it is a number. 
# I encourage you not to distill it to a yes-or-no phrase. 
# p = 0.006 and p = 0.000000006 are both said to be "statistically significant," but they are definitely not the same!

# "HYPOTHESIS TESTING"
# 1) CLEARLY STATE THE NULL HYPOTHESIS
# 2) define your test statistics3)
# 3) generate many sets of simulated data assuming the null hypothesis true
# 4) compute the test statistic for each simulated data set
# 5) the p-value is the fraction of your simulated data sets for which the test statistics
# is at least as extreme as for the real data
# https://github.com/AllenDowney/ThinkStats2/blob/master/thinkstats2/thinkstats2.py#L2987

# A one-sample bootstrap hypothesis test
# GOAL: you want to see if Frog B and Frog C have similar impact forces.

# Another juvenile frog was studied, Frog C, and you want to see if Frog B and Frog C have similar impact forces. 
# Unfortunately, you do not have Frog C's impact forces available, but you know they have a mean of 0.55 N. 
# Because you don't have the original data, you cannot do a permutation test, and you cannot assess the hypothesis that the forces from Frog B and Frog C come from the same distribution. 
# You will therefore test another, less restrictive hypothesis: The mean strike force of Frog B is equal to that of Frog C.

## A one-sample bootstrap hypothesis test
# To set up the bootstrap hypothesis test, you will take the mean as our test statistic. 

# Remember, your goal is to calculate the probability of getting a mean impact force 
# less than or equal to what was observed for Frog B (H0)
# if the hypothesis that the true mean of Frog B's impact forces is equal to that of Frog C is true. (H1)

# You first translate all of the data of Frog B such that the mean is 0.55 N. 
# This involves adding the mean force of Frog C and subtracting the mean force of Frog B from each measurement of Frog B. 
# This leaves other properties of Frog B's distribution, such as the variance, unchanged.

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)
# p =  0.0046
# 0.05%
# Great work! The low p-value suggests that the null hypothesis that Frog B and Frog C have the same mean impact force is false.

#########3


## A two-sample bootstrap hypothesis test for difference of means
# We now want to test the hypothesis that Frog A and Frog B have the same mean impact force, 
# but not necessarily the same distribution, which is also impossible with a permutation test.

# To do the two-sample bootstrap test, we shift both arrays to have the same mean, 
# since we are simulating the hypothesis that their means are, in fact, equal. 
# We then draw bootstrap samples out of the shifted arrays and compute the difference in means. 
# This constitutes a bootstrap replicate, and we generate many of them. 
# The p-value is the fraction of replicates with a difference in means greater than or equal to what was observed.

# The objects forces_concat and empirical_diff_means are already in your namespace.

# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
# SIMULATE HYPOTHESIS THAT THEIR MEANS ARE IN FACT, EQUALS.
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, 10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, 10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b # could be math.abs?

# Compute and print p-value: p
p = np.sum(bs_replicates>= empirical_diff_means) / len(bs_replicates)
print('p-value =', p)
# p-value = 0.0043

# You got a similar result as when you did the permutation test. 
# Nonetheless, remember that it is important to carefully think about what question you want to ask. 
# Are you only interested in the mean impact force, or in the distribution of impact forces?

# Para evaluar la significancia estadística, examine el valor p de la prueba. 
# Si el valor p está por debajo de un nivel de significancia (α) especificado (generalmente 0.10, 0.05 o 0.01), 
# usted puede decir que la diferencia es estadísticamente significativa y rechazar la hipótesis nula de la prueba.