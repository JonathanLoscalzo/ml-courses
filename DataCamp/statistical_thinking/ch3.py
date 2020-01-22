
## Probabilistic logic and statistical inference
# Given a set of data, you describe probabilistically what you might expect 
# if those data were acquired again and over again.
# # Heart of statistical inference, from data to probabilistic conclusions (what you expect if you collect data over and over again)

# Statistical inference involves:
# - taking your data to probabilistic conclusions about what you would expect if you took even more data, 
# - and you can make decisions based on these conclusions.
# 
# Probabilistic language is in fact very precise. 
# It precisely describes uncertainty (incertidumbre).
# ___________________________


# Hacking Statistics
 # uses simulated repeated measurements to compute probabilities. 

# Simulated coin flip with np.random.random() (<0.5, or >= 0.5)

# Probability of obtain 4 heads  (value < 0.5 is head)
# Repeat the experiment over and over again
n_all_heads = 0
experiments = 10000
for _ in range(experiments):
    heads = np.random.random(size=4) < 0.5
    n_heads = np.sum(heads)
    if n_heads == 4: 
        n_all_heads +=1

print(n_all_heads / experiments) # prints probability
# (it's the approximate as 0.5 * 0.5 * 0.5 * 0.5)

# Random probability of obtain heads in coins is 0.5 ( 1 / 2). 
# np.random.random returns real random data between 0 to 1.
# If the numbers are truly random, 
# all bars in the histogram should be of (close to) equal height.

# Specifically, each coin flip has a probability p of landing heads (success) 
# and probability 1−p of landing tails (failure).

# DISCRETE VARIABLES

# Seed the random number generator
np.random.seed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in random_numbers:
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()

# the histogram is almost exactly flat across the top, 
# indicating that there is equal chance that a randomly-generated 
# number is in any of the bins of the histogram.

# _____________________________________


# returns the number of successes out of n Bernoulli trials,
# each of which has probability p of success. 
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number<p:
            n_success +=1

    return n_success


# Include the normed=True keyword argument so that the height of the bars of the histogram indicate the probability.
# normed is deprecated, use density instead
# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

# ______________________________________
# if interest rates are such that the bank will lose money 
# if 10 or more of its loans are defaulted upon, 
# what is the probability that the bank will lose money?

# Compute ECDF: x, y
x,y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
plt.plot(x,y, marker='.', linestyle='none')
plt.xlabel('n_defaults loans')
plt.ylabel('CDF')
# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money=np.sum(n_defaults >=10)

# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))

# As we might expect, we most likely get 5/100 defaults. 
# But we still have about a 2% chance of getting 10 or more defaults out of 100 loans.
# Probability of losing money = 0.022

#______________________
#______________________
#______________________

# Discrete  Uniform PMF
# https://es.wikipedia.org/wiki/Distribuci%C3%B3n_de_probabilidad#Distribuciones_de_variable_discreta
# https://en.wikipedia.org/wiki/Discrete_uniform_distribution

# PMF: set of probabilities of discrete outcomes
# Probability distribution: mathematical description of outcomes
# the discrete uniform distribution is a symmetric probability distribution wherein a finite number of values are equally likely to be observed; every one of n values has equal probability 1/n

# is Binomal Distributed
# - the number of r successes in n Bernoulli trials with probability p of success 
# - the number of r heads in 4 coin flips with probability 0.5 of heads
# np.random.binomial(number_of_bernoulli_trials, probability_of_successes)

# Instead of simulating all of the Bernoulli trials, perform the sampling using np.random.binomial().
# This is identical to the calculation you did 
# in the last set of exercises using your custom-written perform_bernoulli_trials()

# We simulated a story of a person fliping a coin.
# We did this to get the probability of each possible outcome of the story

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n=100,p=0.05,size=10000)

# Compute CDF: x, y
x,y = ecdf(n_defaults)

# Plot the CDF with axis labels
plt.plot(x,y,marker='.', linestyle='none')
plt.xlabel("number of defaults out of 100 loans")
plt.ylabel("CDF")

# Show the plot
plt.show()
#PMF
# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
plt.hist(n_defaults, normed=True, bins=bins)

# Label axes

plt.xlabel('number of defaults out of 100 loans')
plt.ylabel('PMF')

# Show the plot
plt.show()

#_______________________________________________________
# Poisson
# The amount of time you have to wait for a bus is completely independent of when the previous bus arrived
# The timing of the next event is completely independent of when the previous event happened
# Examples:
# -Natural births in a given hospital
# -hit on a website during a given hour
# -meteor strikes
# -molecular collisions in a gas
# -aviation incidents

# Poisson distribution
# Limit of the binomial distribution for low probability of success and a larg number of trials (experiments)
# That is, for rare events

# (np.e ** (-lambda) * lambda**media_events ) / np.math.factorial(media_events) 

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
p = [0.5, 0.1, 0.01]
n = [20, 100, 1000]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))
# Poisson:      10.0186 3.144813832327758
# n = 20 Binom: 9.9637 2.2163443572694206
# n = 100 Binom: 9.9947 3.0135812433050484
# n = 1000 Binom: 9.9985 3.139378561116833

# The means are all about the same, which can be shown to be true by doing some pen-and-paper work. 
# The standard deviation of the Binomial distribution gets closer and closer 
# to that of the Poisson distribution as the probability p gets lower and lower.
# When we have rare events (low p, high n), the Binomial distribution is Poisson.

# 
# When we have rare events (low p, high n), the Binomial distribution is Poisson. 
# This has a single parameter, the mean number of successes per time interval, 
# in our case the mean number of no-hitters per season.

# Was 2015 anomalous?
# 1990 and 2015 featured the most no-hitters of any season of baseball (there were seven). Given that there are on average 251/115 no-hitters per season, what is the probability of having seven or more in a season?
# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters>=7)

# Compute probability of getting seven or more: p_large
p_large = n_large / 10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)

# we are doing "hacking statistics", but the formula was: 
p = lambda l,x : (np.exp(-l) * l ** x ) / math.factorial(x)

print(p(251/115, 7)) # that is equals to 7

# he result is about 0.007. This means that it is not that improbable to see a 7-or-more no-hitter season in a century. We have seen two in a century and a half, so it is not unreasonable.
# https://www.tutorialspoint.com/python_data_science/python_binomial_distribution.htm

# The binomial distribution tends toward the poisson distribution as: 
# n -> infinite
# p -> 0
# lambda = n*p
# see binomial distribution formula https://www.youtube.com/watch?v=ceOwlHnVCqo

# The difference between the two is that while both measure the number of certain random events (or "successes") 
# within a certain frame, the Binomial is based on discrete events, while the Poisson is based on continuous events. 
# That is, with a binomial distribution you have a certain number, n, of "attempts," 
# each of which has probability of success p. With a Poisson distribution, 
# you essentially have infinite attempts, with infinitesimal chance of success. 