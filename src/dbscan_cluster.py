# http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    X = np.array([[20.,20.],[1.,0.],[2.,2.],[1.,1.],[3.,3.],[10.,10.]]*2)
    
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    
    X = StandardScaler().fit_transform(X)
    
    print 'X:',X
    print 'labels_true:',labels_true
    db = DBSCAN(eps=1, min_samples=2).fit(X)
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print db.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print n_clusters_
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
        print X
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def DBCluster( input_list, _eps, _min_samples ):
    
    X = np.array( input_list )
    
    
    db = DBSCAN( eps=_eps, min_samples=_min_samples ).fit(X)
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_    
    
    #print labels
    return labels
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    #unique_labels = set(labels)
    #colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))    
    
    #for k, col in zip(unique_labels, colors):
        #if k == -1:
            ## Black used for noise.
            #col = 'k'
    
        #class_member_mask = (labels == k)
       
        #xy = X[class_member_mask & core_samples_mask]
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 #markeredgecolor='k', markersize=14)
    
        #xy = X[class_member_mask & ~core_samples_mask]
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 #markeredgecolor='k', markersize=6)
    
    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()
if __name__ == '__main__' :
    
    main()