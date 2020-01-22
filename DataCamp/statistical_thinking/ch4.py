# CONTINUOUS VARIABLES
# probability density function PDF
# (similar PMF but for continuous variables)

# It just has to be singularly-valued, nonnegative, and the total area under the PDF must be equal to one.
# NOTA: probability is given by the area under the *PDF, not the value of the PDF.

################## Normal CDF
# CDF show de probability of a variable to have the x-value or less. 
# Example:  97% of the measures observed are light<300.000 km/s
# remember CDF are: X are sort measures, then y are probabilities of have x-value or less. 
# x => sorted(measures)
# y => arange(1, 1+n) / n, where n is len(measures)

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20,1,size=100000)
samples_std3 = np.random.normal(20,3,size=100000)
samples_std10 = np.random.normal(20,10,size=100000)


# Make histograms
plt.hist(samples_std1, normed=True, histtype='step' ,bins=100)
plt.hist(samples_std3, normed=True, histtype='step',bins=100)
plt.hist(samples_std10, normed=True, histtype='step',bins=100)

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()


# Generate CDFs

x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3) 
x_std10, y_std10 = ecdf(samples_std10)


# Plot CDFs
plt.plot(x_std1, y_std1, marker='.', linestyle='none')
plt.plot(x_std3, y_std1, marker='.', linestyle='none')
plt.plot(x_std10, y_std1, marker='.', linestyle='none')


# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()

# the CDFs all pass through the mean at the 50th percentile; 
# the mean and median of a Normal distribution are equal. 
# The width of the CDF varies with the standard deviation.

# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)


# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()

# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples <= 144) / 1000000

# Print the result
print('Probability of besting Secretariat:', prob)
# We get that there is only a 0.06% chance of a horse running the Belmont as fast as Secretariat.

# EXPONENTIAL DISTRIBUTION
# the waiting time between arrivals of a poisson process is exponentially distributed.

# Waiting for the next Secretariat
# Unfortunately, Justin was not alive when Secretariat ran the Belmont in 1973. 
# Do you think he will get to see a performance like that? To answer this, 
# you are interested in how many years you would expect to wait until you see another performance like Secretariat's.
# How is the waiting time until the next performance as good or better than Secretariat's distributed? Choose the best answer.

# Exponential: A horse as fast as Secretariat is a rare event, which can be modeled as a Poisson process, and the waiting time between arrivals of a Poisson process is Exponentially distributed.

#https://en.wikipedia.org/wiki/Exponential_distribution
#How long must we wait to see both a no-hitter and then a batter hit the cycle? 
# The idea is that we have to wait some time for the no-hitter, and then after the no-hitter,
# we have to wait for hitting the cycle. Stated another way, what is the total waiting time for the arrival of two different Poisson processes?
# tau is the mean of poisson process

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)

    return t1 + t2

# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, 100000)

# Make the histogram
plt.hist(waiting_times, bins=100,normed=True, histtype='step')


# Label axes
plt.xlabel("waiting time")
plt.ylabel("PDF")

# Show the plot
plt.show()
#Notice that the PDF is peaked, unlike the waiting time for a single Poisson process. 

