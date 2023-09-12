import scipy.io as scio
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Lasso
import pandas as pd


def main():
    data_file = './data/dblp/DBLP4057_GAT_with_idx.mat'
    data = scio.loadmat(data_file)
    feature = data['features']
    feature = feature * 1.0
    labels = []
    for i in range(data['label'].shape[0]):
        for j in range(data['label'].shape[1]):
            if data['label'][i, j] == 1:
                labels.append(j)
                break
    labels = np.array(labels)
    g = data['net_APA'] - np.eye(labels.shape[0])
    new_edges = []
    for i in range(g.shape[0]):
        for j in range(i+1, g.shape[0]):
            if g[i, j] > 0:
                new_edges.append([i, j])
    new_edges = np.array(new_edges)
    np.savez('./data/dblp/data.npz', ft=feature, lbl=labels, edge=new_edges)


if __name__ == '__main__':
    main()
