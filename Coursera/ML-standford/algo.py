import numpy as np

def entropy(arr):
    counts = np.unique(arr, return_counts = True)
    ar_len = len(arr)
    sz = np.size(counts, axis=1)
    
    return sum([ - (counts[1][i] * np.log2(counts[1][i]/ar_len)) / ar_len  for i in range(sz)])


def gain(arr, divide):
    e = entropy(arr)
    es = [ entropy(i) for i in divide ]
    len_es = [len(i) for i in divide]
    len_e = len(arr)
    return e - sum([s * v / len_e for s,v in zip(es, len_es)])
