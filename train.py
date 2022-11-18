import numpy as np
from cluster import Cluster
from helpers import weight_edges
from scipy.sparse import csr_matrix
from sknetwork.clustering import Louvain
from utils import load_pickle, write_pickle


cluster_manager = Cluster(path='glucose.db', raw=False)
cluster_manager.load_data()

'''
STEP 1: RUN LOUVAIN ALGORITHM
'''
db = load_pickle('data/similar_pairs.dict')
N = len(cluster_manager.events)

alpha, beta = 1/3, 1/3
row, col, data = weight_edges(db, alpha, beta, cluster_manager.cause_effect_id)

similarity_matrix = csr_matrix((data, (row, col)), shape=(N, N))

louvain = Louvain()
labels = louvain.fit_transform(similarity_matrix)

write_pickle(labels, 'models/louvain.cluster')

'''
STEP 2: FINETUNE CLUSTERING RESULTS
'''
import random
from tqdm import tqdm
from cluster import remove_self_loop, event_cluster_similarity, search_candidates

cluster_manager.load_cluster('models/louvain.cluster')

def finetune(cluster_manager, inter = True, intra = True):

  num_events = len(cluster_manager.events)
  
  if inter:
    print('Removing inter-cluster causal relations')
    clusters = []
    for k, cluster in cluster_manager.collector.items():
      print('Processing cluster: ', k)
      shuffled_cluster = cluster.copy()
      random.shuffle(shuffled_cluster)
      new_cluster = remove_self_loop(shuffled_cluster, 
                                    similarity_matrix,
                                    cluster_manager.cause_effect_id)
      assert np.sum([len(v) for k, v in new_cluster.items()]) == len(cluster)
      v = list(new_cluster.values())
      clusters.extend(v)


    expanded_cluster = {i: None for i in range(num_events)}
    for cluster_id, cl in enumerate(clusters):
      for event_id in cl:
        expanded_cluster[event_id] = cluster_id

    updated_cluster = np.array(list(expanded_cluster.values()))
    cluster_manager.update_cluster(updated_cluster)
  
  if intra: 
    print('Removing intra-cluster causal relations')

    M = cluster_manager.extract_relations()
    num_clusters = M.shape[0]

    # Sort by the number of members in a cluster
    sorted_collector = {}
    for k, v in cluster_manager.collector.items():
      sorted_collector[k] = len(v)

      
    sorted_collector = sorted(sorted_collector.items(), key=lambda kv: kv[1], reverse=True)

    updated_collector = {}
    for i, _ in tqdm(sorted_collector):
      updated_collector[i] = [cluster_manager.collector[i]]
      for j in range(num_clusters):
        if i != j and j not in updated_collector:
          
          if min(M[i, j], M[j,i]) > 0:

            ref_cluster = cluster_manager.collector[j]
            new_cluster = []
            for idx, target_cluster in enumerate(updated_collector[i]):
                
              if M[i, j] > M[j, i]:
                # Search member A is a cause of any member B
                removed = search_candidates(target_cluster, ref_cluster, 
                                            'cause', 
                                            cluster_manager.cause_effect_id)
              
              else:
                # Search member A is a effect of any member B
                removed = search_candidates(target_cluster, ref_cluster, 
                                            'effect',
                                            cluster_manager.cause_effect_id)
              
              main = set(target_cluster)
              curr = main - removed
              if len(curr) > 0:
                new_cluster.append(curr)
              if len(removed) > 0:
                new_cluster.append(removed)
              
            
            updated_collector[i] = new_cluster

    c = 0
    for k, v in updated_collector.items():
      cluster_manager.collector[k] = v[0]
      for cl in v[1:]:
        idx = num_clusters + c
        cluster_manager.collector[idx] = cl
        c += 1 
    
    # Already update collector here
    expanded_cluster = {i: None for i in range(num_events)}
    for cluster_id, cl in cluster_manager.collector.items():
      for event_id in cl:
        expanded_cluster[event_id] = cluster_id
    

    updated_cluster = np.array(list(expanded_cluster.values()))
    cluster_manager.cluster = updated_cluster
  
cluster_manager = finetune(cluster_manager)



