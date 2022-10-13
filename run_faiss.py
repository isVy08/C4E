import faiss
import numpy as np
from utils import load_pickle
from helpers import load_data, train_load_embeddings

events, _, _ = load_data(raw=True, path='glucose.db')
train_load_embeddings('data/clean_embeddings.npy', events)

k = 300
X = np.load('data/clean_embeddings.npy')                   
n, d = X.shape
index = faiss.IndexFlatL2(d)   

index.add(X)                  
print('Start searching ...')
D, I = index.search(X, k)
from utils import write_pickle
write_pickle((D,I), 'knn.pairs')
 