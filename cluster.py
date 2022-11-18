import numpy as np
from tqdm import tqdm
from utils import load_pickle
from helpers import check_cause_effect


def event_cluster_similarity(current_cluster, candidate, 
                            similarity_matrix,
                            cause_effect_id):
  s = 0
  for member in current_cluster: 
    if check_cause_effect(candidate, member, cause_effect_id):
      return -1
    else:
      s += similarity_matrix[candidate, member]
  
  return s / len(current_cluster)

def remove_self_loop(cluster, similarity_matrix, cause_effect_id):
  '''
  cluster : a shuffled list of members
  '''
  clusters = {}

  curr = 0
  main = []
  while len(cluster) > 0:
    x = cluster.pop()
    for y in cluster:
      if check_cause_effect(x, y, cause_effect_id):

        cluster.remove(y)
        
        if curr not in clusters: 
          clusters[curr] = {y}
        
        else:
          # check correlation with all clusters from the begining:
          scores = []
          for c in range(curr + 1):
            current_cluster = clusters[c]
            scr = event_cluster_similarity(current_cluster, y, 
                                    similarity_matrix, 
                                    cause_effect_id)
            scores.append(scr)
          
          best_score = np.max(scores)

          if best_score > 0: 
            best_cluster = np.argmax(scores)
            clusters[best_cluster].add(y)
          
          else: 
            curr += 1 
            clusters[curr] = {y}
    
    main.append(x)
  
  clusters[-1] = main
  return clusters


def search_candidates(target_cluster, ref_cluster, 
                      search_type, cause_effect_id):
  
  removed = set()
  if search_type == 'cause':
    # Search member A is a cause of any member B
    for k in ref_cluster:
      try:
        causes = cause_effect_id[k]
        for v in causes: 
          if v in target_cluster:
            removed.add(v)
      except KeyError:
        pass
  
  else:
    # Search member A is a effect of any member B
    for k in target_cluster:
      try:
        causes = cause_effect_id[k]
        for v in causes: 
          if v in ref_cluster:
            removed.add(k)
            break
      except KeyError:
        pass
  
  return removed




class Cluster(object):
  def __init__(self, path='glucose.db', raw=True):
    self.cluster = None
    self.collector = None
  

  def load_data(self, path, raw = True):
    
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

    events, cause_effect, event_loc, cause_effect_id = load_pickle(path)
  
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

    self.events = events 
    self.cause_effect = cause_effect
    self.event_loc = event_loc
    self.cause_effect_id = self.extract_causal_id()

  

  def extract_causal_id(self):
    cause_effect_id = {}
    for k, v in tqdm(self.cause_effect.items()):
      ki = self.events.index(k)
      if ki not in self.cause_effect_id: 
        cause_effect_id[ki] = set()
      
      for ev in v: 
        vi = self.events.index(ev)
        cause_effect_id[ki].add(vi)
    
    return cause_effect_id

  def update_cluster(self, cluster):
    
    self.cluster = cluster
    self.collector = self.group_by_cluster(self.cluster)

  def load_cluster(self, path):
    '''
    cluster : np.array, assignments of each event
    '''
    self.cluster = load_pickle(path)
    self.collector = self.group_by_cluster(self.cluster)


  def group_by_cluster(self, cluster):
    collector = {}
    for event_id, cluster_id in enumerate(cluster):
      if cluster_id not in collector:
        collector[cluster_id] = []
      
      collector[cluster_id].append(event_id)
    
    return collector

  def extract_relations(self):  
    
    num_clusters = len(self.collector)
    
    # how many times cluster i has a member that is a cause of any member of cluster j
    M = np.zeros(shape=(num_clusters, num_clusters)) 

    for cluster_id, cl in self.collector.items():
      for event_id in cl:
        if event_id in self.cause_effect_id:
          causes = self.cause_effect_id[event_id]
          for cau_event_id in causes:
            parent_cluster_id = self.cluster[cau_event_id]
            M[parent_cluster_id, cluster_id] += 1
      
    return M
  
  def evaluate(self):
    num_clusters = len(self.collector)
    M = self.extract_relations()
    sparsity = 1.0 - ( np.count_nonzero(M) / float(M.size) )
    
    # bidirectional ratio
    bdr = self.bidir_ratio(M)
    
    # self-loop ratio
    arr = [len(cl) for _, cl in self.collector.items()]
    diag_ = M.diagonal() // 2
    slr = np.divide(diag_, arr).mean()
    
    max_cluster_size = max(arr) / self.cluster.shape[0]
    print('Number of clusers:', num_clusters)
    print('Bidirectional Ratio:', bdr)
    print('Self-loop Ratio:', slr)
    print('Sparsity:', sparsity)
    print('Max cluster size:', max_cluster_size)
    return M 

  def bidir_ratio(self, M):
    bdr = 0
    cnt = 0
    n = M.shape[0]
    for i in range(n-1):
      for j in range(i+1, n):
        ce = [M[i,j], M[j, i]]
        if max(ce) != 0:
          bdr += min(ce) / max(ce)
          cnt += 1
    
    cnt = max(1, cnt)
    return bdr/cnt


  def view_cluster(self, cluster_index, first=20):
    for m in self.collector[cluster_index][:first]:
      print(self.events[m])

  