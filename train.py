from utils import load_pickle
from cluster import Cluster
from scipy.sparse import csr_matrix
from sknetwork.clustering import Louvain



def euclidean_similarity(dist, kernel = 10):
  denom = 2 * kernel**2
  return np.exp(-dist / denom)
  
def check_cause_effect(x, y, cause_effect_id):
    if cause_effect_id is None:
        return False

    if x in cluster_manager.cause_effect_id:
        if y in cluster_manager.cause_effect_id[x]:
        return True

    if y in cluster_manager.cause_effect_id:
        if x in cluster_manager.cause_effect_id[y]:
        return True
    
    return False

def weight_edges(db, w_pp, w_iou, cause_effect_id):
    '''
    generate edge weights for the adjacency matrix
    db : dict of pairwise scores, for example (0, 1202) : {'pp': 0.80, 'iou' : 0.5, 'dist': 1.0} 
    where dist is the Euclidean distance
    '''
    row = []
    col = []
    data = []

    for k, v in tqdm(db.items()):
        x, y = k
    
    if check_cause_effect(x, y, cause_effect_id):
        s = 0.0
    else:
        sim = euclidean_similarity(v['dist'], kernel = 10)
        s = w_pp * v['pp'] + w_iou * v['iou'] + (1 - w_pp - w_iou) * sim

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

if __name__ == "__main__":

    cluster_manager = Cluster(path='glucose.db', raw=False)
    db = load_pickle('data/similar_pairs.dict')
    N = len(cluster_manager.events)

    w_pp, w_iou = 1/3, 1/3
    row, col, data = weight_edges(db, w_pp, w_iou, cluster_manager.cause_effect_id)

    adjacency = csr_matrix((data, (row, col)), shape=(N, N))

    louvain = Louvain()
    labels = louvain.fit_transform(adjacency)

    write_pickle(labels, 'models/louvain.cluster')


