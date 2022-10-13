import os, time
import numpy as np
from tqdm import tqdm
from helpers import load_data
from sklearn import metrics
from utils import load_pickle, write_pickle

class Cluster(object):
  def __init__(self, path='glucose.db', raw=True):
    self.events, self.cause_effect, self.event_loc = load_data(raw, path)
    self.cluster = None
    self.collector = None
  

  def extract_causal_id(self):
    self.cause_effect_id = {}
    for k, v in tqdm(self.cause_effect.items()):
      ki = self.events.index(k)
      if ki not in self.cause_effect_id: 
        self.cause_effect_id[ki] = set()
      
      for ev in v: 
        vi = self.events.index(ev)
        self.cause_effect_id[ki].add(vi)

  
  def load_cluster(self, path):
    '''
    returns: list, cluster id assigned to each event
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
    events_dict = {event: i for i, event in enumerate(self.events)}
     
    nc = len(self.collector)
    
    # how many times a member of cluster i is a cause of any member of cluster j
    M = np.zeros(shape=(nc, nc)) 

    for cluster_id, cl in self.collector.items():
      nm = len(cl)
      for event_id in cl:
        event = self.events[event_id]
        if event in self.cause_effect:
          causes = self.cause_effect[event]
          for cau_event in causes:
            try:
              cau_event_id = events_dict[cau_event]
              if cau_event_id < nc:
                # find parent cluster
                parent_cluster_id = self.cluster[cau_event_id]
                if parent_cluster_id < nc:
                  M[parent_cluster_id, cluster_id] += 1
            except KeyError:
              pass
      
    return M
  
  def evaluate(self, cluster):
    n_clusters = cluster.max() + 1
    M = self.extract_relations(cluster)
    arr = [len(cl) for _, cl in self.collector.items()]
    diag_ = M.diagonal() // 2
    inner = np.divide(diag_, arr).mean()
    sparsity, outer = self.relation_quality(M)
    balance = max(arr) / cluster.shape[0]
    return n_clusters, sparsity, outer, inner, balance

  def relation_quality(self, M):
    sparsity = 1.0 - ( np.count_nonzero(M) / float(M.size) )
    outer = 0
    cnt = 0
    n = M.shape[0]
    for i in range(n-1):
      for j in range(i+1, n):
        ce = [M[i,j], M[j, i]]
        if max(ce) != 0:
          outer += min(ce) / max(ce)
          cnt += 1
    if cnt == 0:
      cnt == 1
    return sparsity, outer/cnt


  def view_cluster(self, cluster_index, first=20):
    for m in self.collector[cluster_index][:first]:
      print(self.events[m])

  