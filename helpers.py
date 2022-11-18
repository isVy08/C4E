import numpy as np
from tqdm import tqdm
import scipy, os, torch


'''
COMPUTE EVENT SIMILARITY FOR CORRELATION CLUSTERING
'''

def train_load_embeddings(path, events=None):
  
  
  if os.path.isfile(path):
    print('Loading pre-trained embeddings ...')
    corpus_embeddings = np.load(path)

  else:
    assert events is not None, 'Event data is required!'
    print('Extracting embeddings ...')
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models')
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(events, batch_size=256, show_progress_bar=True, convert_to_tensor=False)
    print('Encoding finished!') 
    np.save(path, corpus_embeddings)

  print(corpus_embeddings.shape)
  return corpus_embeddings


def load_transformer(usage):
  if usage == 'phr':
    # Load paraphrase detector model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("coderpotter/adversarial-paraphrasing-detector")
    model = AutoModelForSequenceClassification.from_pretrained("coderpotter/adversarial-paraphrasing-detector")
    if torch.cuda.is_available():
      model.to('cuda')
  elif usage == 'nli': 
    # Load NLI model
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/nli-roberta-base')
    tokenizer = None
  
  return model, tokenizer


def score_phr(batch, tokenizer, phr_model):
  inputs = tokenizer(batch, padding=True, truncation=True)
  for k, v in inputs.items():
    inputs[k] = torch.Tensor(v).long()
    if torch.cuda.is_available():
      inputs[k] = inputs[k].to('cuda')
      
  scr = phr_model(**inputs).logits
  scr = scr.detach().cpu().numpy()
  del inputs
  return scr

def generate_input_batch(events, x, y_batch, usage):
  input = []
  for y in y_batch:
    if usage == 'nli':
      input.append((events[x], events[y]))
    elif usage == 'phr':
      s = events[x] + '. ' + events[y] + '.'
      input.append(s)
    else: 
      raise ValueError('usage takes either "phr" or "nli"')  
  return input
  

def pairwise_nli_phr(events, pairs, 
                batch_size, usage,
            
                threshold=0.1, path = None):
  '''
  events    : list, of text events
  pairs     : dict, key: list of neighbors (filterd by using other similarity metrics)
  batch_size: int
  usage     : str, either "phr" or "nli
  threshold : min similarity score to write 
  path      : str, saved file name
  
  output written into <path> file, each line is a tuple (1st-event-id, 2nd-event-id, similarity-score)
  '''
  if path is None:
    path = f'data/{usage}_similarity.txt'

  model, tokenizer = load_transformer(usage)
  
  N = len(events)
  file = open(path, 'a+')
  print('Start processing ...') 
  for x in tqdm(range(N)):
    if x in pairs:
      nb = list(pairs[x]) 
      for i in range(0, len(nb), batch_size):
        y_batch = nb[i: i + batch_size]
        input = generate_input_batch(events, x, y_batch, usage)
        
        if usage == 'nli':
          scr = model.predict(input) 
        elif usage == 'phr':
          scr = score_phr(input, tokenizer, model)
        else:
          raise ValueError('usage takes either "phr" or "nli"')
        
        probs = scipy.special.softmax(scr,-1)[:, 1]
        locations = np.where(probs > threshold)[0]
        
        for l in locations:
          y = y_batch[l]
          p = probs[l]
          msg = (x, y, p)
          file.write(str(msg) + '\n')

        
  file.close()
  print(f'Output is written at {path}')



def pairwise_iou(events, 
                threshold=0.4, 
                path='data/iou_similarity.txt'):
  
  import string
  def tokenizer(event):
    out = []
    token_list = event.split(' ')
    for tok in token_list: 
      if tok not in string.punctuation and len(tok) > 0:
        out.append(tok)
    return set(out) 

  print('Tokenizing data ...')
  tokens = []
  for ev in tqdm(events):
    token_set = tokenizer(ev)
    tokens.append(token_set)
  
  print('Start processing ...')
  file = open(path, 'a+')
  N = len(tokens)
  for x in tqdm(range(N-1)):
    tx = tokens[x]
    if len(tx) > 0:
      for y in range(x+1, N):
        ty = tokens[y]      
        if len(ty) > 0:
          iou = len((tx & ty)) / len((tx | ty)) 
          if iou >= threshold:
            msg = (x, y, iou)
            file.write(str(msg) + '\n')
    
  file.close()
  print(f'Output is written at {path}')

def weight_edges(db, alpha, beta, cause_effect_id, penalty = 0):
    '''
    generate edge weights for the adjacency matrix
    db : dict of pairwise scores, for example (0, 1202) : {'phr': 0.80, 'iou' : 0.5, 'dist': 1.0} 
    where pp is paraphrase likelihood, iou is IoU, dist is the Euclidean distance
    '''
    row = []
    col = []
    data = []

    for k, v in tqdm(db.items()):
        x, y = k
    
        if check_cause_effect(x, y, cause_effect_id):
            s = penalty
        else:
            euclidean_sim = 1 / (1 + v['dist'])
            s = alpha * v['phr'] + beta * v['iou'] + (1 - alpha - beta) * euclidean_sim

        row.append(x)
        row.append(y)
        
        col.append(y)
        col.append(x)
        
        data.append(s)
        data.append(s)

    row = np.array(row) 
    col = np.array(col) 
    data = np.array(data)
    print(row.shape, col.shape, data.shape)
    return row, col, data


def check_cause_effect(x, y, cause_effect_id):
    if cause_effect_id is None:
        return False

    if x in cause_effect_id:
        if y in cause_effect_id[x]:
          return True

    if y in cause_effect_id:
        if x in cause_effect_id[y]:
          return True
    
    return False