# Non Negative matrix factorization
# NMF: non negative matrix factorization
# Other reduction dimension technique, (dimensionality reduction, source separation or topic extraction.)
# NMF models are interpretable (unlike PCA)

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

# - reconstruct sample by multiply components by feature values, and adding up
# it only be applied to arrays of non-negative data such us
# - word frequencies arrays
# - image encoded as arrays
# - audio spectrograms
# - purchase e-commerce

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])
# Name: Anne Hathaway, dtype: float64
# <script.py> output:
#     0    0.003845
#     1    0.000000
#     2    0.000000
#     3    0.575711
#     4    0.000000
#     5    0.000000
# Name: Anne Hathaway, dtype: float64
#     0    0.000000
#     1    0.005601
#     2    0.000000
#     3    0.422380
#     4    0.000000
#     5    0.000000
#     Name: Denzel Washington, dtype: float64

import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

# np.dot(W,H) similar => X

#######################33
### NMF learn interpertable parts
# for documents: 
#     NMF components represent topics
#     & NMF features combine topics into documents
# for images: 
#         nmf components are parts of images    

# NMF is applied to documents, the components correspond to topics of documents, 
# and the NMF features reconstruct the documents from the topics. 

# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())
# film       0.627877
# award      0.253131
# starred    0.245284
# role       0.211451
# actress    0.186398
# Take a moment to recognise the topics that the articles about Anne Hathaway and Denzel Washington have in common!

# hay que ver lo siguiente: 
# # non-negative matrix factorization
# W * H  = V => v original matrix
# H te dice el peso de cada componente (las filas)
# Las columnas te indican el peso de cada palabra en ese componente. 

# En el ejemplo se ve que Anne y Denzel tienen un peso alto en el componente 3, lo buscamos y vemos cuales son las palabras más relevantes


# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

##################


def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0]

# Print digit_features
print(digit_features)

# number 7 is expressed as 3 components
# initial: shape. (100, 104)
# after transformation: shape (100, 7)
# (reduce 104 columns into 7)

# Unlike NMF, PCA doesn't learn the parts of things. 
# Its components do not correspond to topics (in the case of documents) 
# or to parts of images, when trained on images

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Notice that the components of PCA do not represent meaningful parts of images of LED digits!

############
Recommender System
-Apply NMF to word frequency.
-NMF feature values describe topics
-Similar documents have similar topics

- compare with cosine similarity: max:1, min: 0. 
- normalize NFM features
##########
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
#####
# Cristiano Ronaldo                1.000000
# Franck Ribéry                    0.999972
# Radamel Falcao                   0.999942
# Zlatan Ibrahimović               0.999942
# France national football team    0.999923

############################
#################3
# You are given a sparse array artists whose rows correspond to artists 
# and whose column correspond to users. The entries give the number of times 
# each artist was listened to by each user.

# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components = 20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
similarities.nlargest()

###########
# Bruce Springsteen    1.000000
# Neil Young           0.955896
# Van Morrison         0.872452
# Leonard Cohen        0.864763
# Bob Dylan            0.859047

# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
