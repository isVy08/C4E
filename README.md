# C4E
Causal Discovery for Events Flow

## Dependencies
```
pip install -r -requirements.txt
python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")
```

## Data
`preprocess.py` downloads [Glucose dataset](https://huggingface.co/datasets/glucose), extracts separate *cause* and *effect* expressions and remove duplicates.    
Pre-processed data `glucose.db` can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1eKV4ehdlZaR4NOq4mn7iUsSQi9xtG1bS?usp=sharing).

## Clustering
Call the `Cluster` object where `raw=False` will activate text cleansing beforehand. 
```
from cluster import Cluster
cluster_manager = Cluster(path='glucose.db', raw=False)
```

`run_faiss.py` obtains SBERT sentence embeddings and find top *k* nearest neighbors using [FAISS](https://github.com/facebookresearch/faiss).
Note that FAISS computes Euclidean distance. It returns neigboring indices and corresponding distances for each instance in Numpy matrices. 

`helpers.py` provides code for extracting pairwse (1) string similarity IoU and (2) paraphrasing probability from [paraphrase detection model](https://huggingface.co/coderpotter/adversarial-paraphrasing-detector).  

The curated pairs `similar_pairs.dict` can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1eKV4ehdlZaR4NOq4mn7iUsSQi9xtG1bS?usp=sharing). It is a Python dictionary where key is tuple pair and value is a dictionary containing pairwise scores, e.g., 
`(0, 1202) : {'pp': 0.80, 'iou' : 0.5, 'dist': 1.0}`. `dist` is the Euclidean distance. 

Refer to `train.py` on how to perform clustering. 

