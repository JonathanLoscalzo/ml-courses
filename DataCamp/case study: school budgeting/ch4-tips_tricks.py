#
# text:
# NLP tricks
# - tokenize
# - include ngram, 1-ngram, 2-gram...
# e.g. CountVectorizer(ngram_range=(1,2))

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the text vector
text_vector = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit text_features to the text vector
text_features.fit(text_vector)

# Print the first 10 tokens
print(text_features.get_feature_names()[:10])

###############################

# Special functions: You'll notice a couple of new steps provided in the pipeline in this and many of the remaining exercises. Specifically, the dim_red step following the vectorizer step , and the scale step preceeding the clf (classification) step.
# These have been added in order to account for the fact that 
# you're using a reduced-size sample of the full dataset in this course. 
# To make sure the models perform as the expert competition winner intended, 
# we have to apply a dimensionality reduction technique, 
# which is what the dim_red step does, and we have to scale the 
# features to lie between -1 and 1, which is what the scale step does

# The dim_red step uses a scikit-learn function called SelectKBest(), applying something called the chi-squared test to select the K "best" features. 
# The scale step uses a scikit-learn function called MaxAbsScaler() in order to squash the relevant features into the interval -1 to 1.

# https://en.wikipedia.org/wiki/Dimensionality_reduction
# https://en.wikipedia.org/wiki/Chi-squared_test
# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    (
                        'vectorizer', 
                        CountVectorizer(
                            token_pattern=TOKENS_ALPHANUMERIC,
                            ngram_range=(1,2))
                    ),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

########################
# stats trick
# interaction terms: 
# what if terms are not next to each other? (que pasa si los términos no están unos pegados al otro?)
# - english teacher - 2nd grade

# beta1 * x1 + beta2 * x2 + beta3 * (x1 * x2)
# x1 = 0/1, x2 = 0/1 => third term only appear if x1 and x2 are one. (both occur)
# beta3: how important is x1,x2 appear together

from sklearn.preprocessing import PolynomialFeatures
# params: - degree, include_bias, interaction_only

# SparseInteractions is here: https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/features/SparseInteractions.py

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),  
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

####################
# Computational trick: hashing
# more features, more memory!
# we want to make an array of features as small as possible
# - dimensionality reduction 
# - in text is very common and useful!

from sklearn.feature_extraction.text import HashingVectorizer
help(HashingVectorizer)
vec = HashingVectorizer(
    norm=None, 
    non_negative=True,  #(?)
    token_pattern=TOKENS_ALPHANUMERIC, 
    ngram_range=(1,2)
)

# tricks 
# NLP: range n-grams, tokenization, 
# stats: interaction terms
# computation: hashing vectorizer. 

#As you saw in the video, HashingVectorizer acts just 
# like CountVectorizer in that it can accept token_pattern 
# and ngram_range parameters. The important difference is 
# that it creates hash values from the text, so that 
# we get all the computational advantages of hashing!

# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Get text data: text_data
text_data = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 

# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(
   # norm=None, 
   # non_negative=True,  #(?)
    token_pattern=TOKENS_ALPHANUMERIC, 
   # ngram_range=(1,2)
)

# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)

# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())

########################

# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(
                        token_pattern=TOKENS_ALPHANUMERIC, 
                        non_negative=True, norm=None, 
                        binary=False,
                        ngram_range=(1,2)
                    )),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Well done! Log loss: 1.2258. Looks like the performance is about the same, 
# but this is expected since the HashingVectorizer should work 
# the same as the CountVectorizer.

# https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/notebooks/1.0-full-model.ipynb
