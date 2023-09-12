import scipy.io as scio
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Lasso
import pandas as pd


def lasso_measure(x, y):
    lasso = Lasso(alpha=0.001, random_state=1)
    lasso.fit(x, y)
    # lasso.fit([[1,2,3],[2,3,1],[3,1,2]],[0,1,1])
    # lasso.fit([[1, 2], [2, 3], [3, 4]], [100,101,102])
    # print(lasso.coef_.tolist())
    imp = []
    for i in range(x.shape[1]):
        if lasso.coef_.tolist()[i] != 0:
            imp.append(i)
    x = x[:, np.array(imp)]
    return x


def gen_knn_edge(X, k):
    n_nodes = X.shape[0]
    chk = np.zeros((n_nodes, n_nodes))
    m_dist = pairwise_distances(X)
    m_neighbors = np.argpartition(m_dist, kth=k + 1, axis=1)
    m_neighbors = m_neighbors[:, :k + 1]
    res_edge = []
    for i in range(n_nodes):
        for j in range(k + 1):
            if i == m_neighbors[i, j]:
                continue
            if chk[i, m_neighbors[i, j]] == 1 or chk[m_neighbors[i, j], i] == 1:
                continue
            res_edge.append([i, m_neighbors[i, j]])
            chk[i, m_neighbors[i, j]] = 1
            chk[m_neighbors[i, j], i] = 1
    return np.array(res_edge)


def main():
    data_file = './data/SFCMatrixZ/SFCMatrixZ.mat'

    data = scio.loadmat(data_file)
    res = np.array(data['data'])
    res = res[:, 1:]
    lbl = np.concatenate((np.ones(76), np.zeros(91)))

    idx = np.arange(lbl.shape[0])
    np.random.shuffle(idx)
    res = res[idx]
    lbl = lbl[idx]
    res = lasso_measure(res,lbl)
    edge = gen_knn_edge(res, k=5)
    np.savez('./data/SFCMatrixZ/data.npz', ft=res, lbl=lbl, edge=edge)


if __name__ == '__main__':
    main()
