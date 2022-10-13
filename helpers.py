import itertools
import numpy as np
import os, torch, scipy
from tqdm import tqdm
from utils import load, load_pickle, write_pickle


def load_data(raw=True, path='glucose.db'):

  def clean_single_event(ev):
    tokens = ev.lower().split(' ')
    tokens = [tok for tok in tokens if 'some' not in tok]
    ev = ' '.join(tokens)
    return ev


  def clean_events(events):
    clean_events = []
    for ev in events: 
      ev = clean_single_event(ev)
      clean_events.append(ev)
    return clean_events

  events, cause_effect, event_loc = load_pickle(path)
  # stories = list(itertools.chain(*event_loc.values()))
  if not raw: 
    events = clean_events(events)
    clean_cause_effect = {}
    for k, v in tqdm(cause_effect.items()):
      k = clean_single_event(k)
      if k not in clean_cause_effect:
        clean_cause_effect[k] = set()
      
      for ev in v:
        ev = clean_single_event(ev)
        clean_cause_effect[k].add(ev) 

    cause_effect = clean_cause_effect

  print(len(events), len(cause_effect), len(event_loc))
  return events, cause_effect, event_loc
  

def train_load_embeddings(path, events=None):
  if os.path.isfile(path):
    print('Loading embeddings ...')
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

# Load paraphrase detector model
def load_transformer(usage):
  if usage == 'pp':
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("coderpotter/adversarial-paraphrasing-detector")
    model = AutoModelForSequenceClassification.from_pretrained("coderpotter/adversarial-paraphrasing-detector")
    if torch.cuda.is_available():
      model.to('cuda')
  elif usage == 'nli': 
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/nli-roberta-base')
    tokenizer = None
  
  return model, tokenizer

def extract_iou_pairs(events, threshold=0.4, 
                      path='data/iou_pairs.txt'):
  
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
  
  print('Extraction begins ...')
  file = open(path, 'a+')
  N = len(tokens)
  for i in tqdm(range(N-1)):
    ti = tokens[i]
    if len(ti) > 0:
      for j in range(i+1, N):
        tj = tokens[j]      
        if len(tj) > 0:
          iou = len((ti & tj)) / len((ti | tj)) 
          # print(iou)
          if iou >= threshold:
            msg = (i, j, iou)
            file.write(str(msg) + '\n')
    
  file.close()


def score_pp(batch, tokenizer, pp_model):
  inputs = tokenizer(batch, padding=True, truncation=True)
  for k, v in inputs.items():
    inputs[k] = torch.Tensor(v).long()
    if torch.cuda.is_available():
      inputs[k] = inputs[k].to('cuda')
      
  scr = pp_model(**inputs).logits
  scr = scr.detach().cpu().numpy()
  del inputs
  return scr

def generate_input_batch(events, id_batch, method):
  input = []
  for y in id_batch:
    if method == 'nli':
      input.append((events[x], events[y]))
    elif method == 'pp':
      s = events[x] + '. ' + events[y] + '.'
      input.append(s)
  
  return input
  

def score_pairs(events, pairs, batch_size, method,
                model, tokenizer=None, threshold=0.1):

  
  
  path = f'data/{method}_score.txt'
  N = len(events)
  file = open(path, 'a+')
  print('Start processing ...') 
  for x in tqdm(range(N)):
    try:
      v = pairs[x] 
      n = len(v)
      for j in range(0, n, batch_size):
        batch = list(v)[j: j + batch_size]
        input = generate_input_batch(events, batch, method)
        
        if method == 'nli':
          scr = model.predict(input) 
        elif method == 'pp':
          scr = score_pp(input, tokenizer, model)
        
        probs = scipy.special.softmax(scr,-1)[:, 1]
        locations = np.where(probs > threshold)[0]
        
        for l in locations:
          y = batch[l]
          p = probs[l]
          msg = (x, y, p)
          file.write(str(msg) + '\n')

    except KeyError:
      pass
        
  file.close()

