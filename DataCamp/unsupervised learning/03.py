###########
# # Dimension Reduction
# - more efficient storage & computation
# - remove less-informative noise features
# - which cause problems for prediction tasks, classification, regression

# PCA
# first: decorrelation (rotation) => 
# second: reduce dimention

# shifts data so they have mean 0.  (no information is lost)
# resulting pca features are not linearly correlated (decorrelated, o r rotated)
# Linear correlation could be measure with pearson correlation.

# principal-components of the data: PCA align with the axes.
# - EACH row defines displacement of the mean (desplazamiento de la media de datos)


# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)

# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)


#######################
# # Intrinsic dimension of a flight path
# Number of features needed to approximate the dataset
# "What is the most compact representation of the samples?""

# intrinsic dimension: number of PCA features with significant variance (high variance)
# if we plot pca features by variance, last features have low variance.
# (Inspecting the scatter plot of features too)
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
############ Found intrinsic PCA features
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

###########

## Dimension Reduction
# represent same data using less features, 
# important if we want to reduce features without loss information.
# assume the high variance features are informative

# word frequency arrays: 
# each index document, each col a word, cell: tf-idf
# Array is "sparse": most entries are 0
# scipy.sparse.csr_matrix instead np.array: (save)

# PCA doesn't support sparse matrix, use TruncatedSVD instead
# TruncatedSVD performs same transformation  but accept sparse matrix as an input

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components = 2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

# In this exercise, you'll create a tf-idf word frequency array for a toy collection of documents. 
# For this, use the TfidfVectorizer from sklearn.
#  It transforms a list of documents into a word frequency array, 
# which it outputs as a csr_matrix. It has fit() and transform() methods like other sklearn objects.

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)

# You saw in the video that TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, 
# such as word-frequency arrays. Combine your knowledge of TruncatedSVD and k-means to cluster 
# some popular pages from Wikipedia. In this exercise, build the pipeline. 
# In the next exercise, you'll apply it to the word-frequency array of some Wikipedia articles.

# Create a Pipeline object consisting of a TruncatedSVD followed by KMeans. 
# (This time, we've precomputed the word-frequency matrix for you, so there's 
# no need for a TfidfVectorizer).

# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values(by='label'))






