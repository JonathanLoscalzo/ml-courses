from gensim import corpora, models, similarities
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

texts = [
    "I love reading Japanese novels. My favorite Japanese writer is Tanizaki Junichiro.",
    "Natsume Soseki is a well-known Japanese novelist and his Kokoro is a masterpiece.",
    "American modern poetry is good. ",
]

keyword = "Japan has some great novelists. Who is your favorite Japanese writer?"

texts = [tokenizer.tokenize(text) for text in texts]
print(texts)

dictionary = corpora.Dictionary(texts)
print(dictionary, dir(dictionary))

feature_cnt = len(dictionary.token2id)
print(feature_cnt)

corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)

tfidf = models.TfidfModel(corpus)
print(tfidf, dir(tfidf)) 

kw_vector = dictionary.doc2bow(tokenizer.tokenize(keyword))
print(kw_vector)

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)

sim = index[tfidf[kw_vector]]

for i in range(len(sim)):
    print('keyword is similar to text%d: %.2f' % (i + 1, sim[i]))