import spacy, nlkt
from utils import write_pickle
from datasets import load_dataset, concatenate_datasets


def preprocess(text):
  doc = nlp(text)
  clean_text = []
  for token in doc:
    lemma = token.lemma_.strip()
    if len(lemma) > 0:
      clean_text.append(lemma)
  
  return ' '.join(clean_text)

def generate_data(dataset):
  '''
  events       : set, of pre-processed text events
  cause_effect : dict, mapping an effect text event to a set of cause text event
  event_loc    : dict, mapping a text event to a set of story ids, used to trace co-occurrence
  '''

  events, event_loc = set(), {}
  cause_effect = {} # effect : cause
  for story in tqdm(dataset):
      ce = story['1_generalNL']
      if ce not in ['escaped', 'answered']:

        c, e = ce.split(' >Causes/Enables> ')
        c = preprocess(c)
        e = preprocess(e)
        
        if e not in cause_effect:
          cause_effect[e] = set()
        
        cause_effect[e].add(c)

        for event in (c, e):
          if event not in event_loc: 
            event_loc[event] = set()

          event_loc[event].add(story['story_id'])
          events.add(event)
        
      
  return events, cause_effect, event_loc



dataset_split = load_dataset('glucose', cache_dir='data')
dataset = concatenate_datasets([dataset_split['train'], dataset_split['test']])
events, cause_effect, event_loc = generate_data(dataset)
write_pickle((events, cause_effect, event_loc), 'glucose.db')