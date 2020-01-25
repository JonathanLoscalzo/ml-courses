# A/B testing
# he Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. 
# Excluding "present" and "abstain" votes, 153 House Democrats and 136 Republicans voted yea. 
# However, 91 Democrats and 35 Republicans voted nay. 
# Did party affiliation make a difference in the vote?

# To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or her vote. 
# You will use the fraction of Democrats voting in favor as your test statistic 
# and evaluate the probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244. 
# (That's right, at least as small as. 
# In 1964, it was the Democrats who were less progressive on civil rights issues.) 
# To do this, permute the party labels of the House voters and then arbitrarily divide them into "Democrats" and "Republicans" and compute the fraction of Democrats voting yea.

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems) / np.size(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)

# Great work! This small p-value suggests that party identity had a lot to do with the voting. 
# Importantly, the South had a higher fraction of Democrat representatives, and consequently also a more racist bias.


#A time-on-website analog


# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, size=10000)


# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) / len(perm_replicates)
print('p-val =', p)

# Your p-value is 0.0001, which means that only one out of your 10,000 replicates had a result as extreme as the actual difference between the dead ball and live ball eras. 
# This suggests strong statistical significance. Watch out, though, you could very well have gotten zero replicates that were as extreme as the observed value. 
# This just means that the p-value is quite small, almost certainly smaller than 0.001.

#TEST OF CORRELATION
# hypothesis test of CORRELATION
# -posit null hypothesis: the thow variables are completely uncorrelated
# -simulate data assuming null hypothesis is True
# -use pearson correlation as test statistics
# -compute p-value as fraction of replicates that have p at least as large as observed.


# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] =  pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates>=r_obs) / len(perm_replicates)
print('p-val =', p)
# p tends to be 0


# You got a p-value of zero. In hacker statistics, this means that your p-value is very low, 
# since you never got a single replicate in the 10,000 you took that had a Pearson correlation greater than the observed one. 
# You could try increasing the number of replicates you take to continue to move the upper bound on your p-value lower and lower.


# Compute x,y values for ECDFs
x_control, y_control = ecdf(control)
x_treated, y_treated = ecdf(treated)

# Plot the ECDFs
plt.plot(x_control, y_control, marker='.', linestyle='none')
plt.plot(x_treated, y_treated, marker='.', linestyle='none')

# Set the margins
plt.margins(0.02)

# Add a legend
plt.legend(('control', 'treated'), loc='lower right')

# Label axes and show plot
plt.xlabel('millions of alive sperm per mL')
plt.ylabel('ECDF')
plt.show()

# Nice plot! The ECDFs show a pretty clear difference between the treatment and control; 
# treated bees have fewer alive sperm. Let's now do a hypothesis test in the next exercise.

# Bootstrap hypothesis test on bee sperm counts
# Now, you will test the following hypothesis: 
# On average, male bees treated with neonicotinoid insecticide have the same number of active sperm per milliliter of semen than do untreated male bees.
#  You will use the difference of means as your test statistic.

# Compute the difference in mean sperm count: diff_means
diff_means = diff_of_means(control, treated)

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate([control, treated]))

# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)

# Nice work! The p-value is small, most likely less than 0.0001, since you never saw a bootstrap replicated with a difference of means at least as extreme as what was observed. 
# In fact, when I did the calculation with 10 million replicates, I got a p-value of 2e-05.
