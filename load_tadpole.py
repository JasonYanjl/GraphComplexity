import scipy.io as scio
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.spatial import distance
import sys
import csv
import pandas as pd
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_tadpole_data(sparsity_threshold):
    with open('./data/tadpole/tadpole_2.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        apoe = []
        ages = []
        gender = []
        fdg = []
        features = []
        labels = []
        cnt = 0
        apoe_col_num = 0
        age_col_num = 0
        gender_col_num = 0
        fdg_col_num = 0
        label_col_num = 0
        for row in rows:
            if cnt != 0:
                row_features = row[fdg_col_num + 1:]
                if row_features.count('') == 0 and row[apoe_col_num] != '':
                    apoe.append(int(row[apoe_col_num]))
                    ages.append(float(row[age_col_num]))
                    gender.append(row[gender_col_num])
                    fdg.append(float(row[fdg_col_num]))
                    labels.append(int(row[label_col_num]) - 1)
                    features.append([float(item) for item in row_features])
                    cnt += 1
            else:
                apoe_col_num = row.index('APOE4')
                age_col_num = row.index('AGE')
                gender_col_num = row.index('PTGENDER')
                fdg_col_num = row.index('FDG')
                label_col_num = row.index('DXCHANGE') # 1->normal 2->MCI 3->AD
                cnt += 1

        num_nodes = len(labels)

        apoe_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if apoe[i] == apoe[j]:
                    apoe_affinity[i, j] = apoe_affinity[j, i] = 1

        age_threshold = 2
        age_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.abs(ages[i] - ages[j]) <= age_threshold:
                    age_affinity[i, j] = age_affinity[j, i] = 1

        gender_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if gender[i] == gender[j]:
                    gender_affinity[i, j] = gender_affinity[j, i] = 1

        reshaped_fdg = np.reshape(np.asarray(fdg), newshape=[-1, 1])
        svc = svm.SVC(kernel='linear').fit(reshaped_fdg, labels)
        prediction = svc.predict(reshaped_fdg)
        fdg_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if prediction[i] == prediction[j]:
                    fdg_affinity[i, j] = fdg_affinity[j, i] = 1

        features = np.asarray(features)
        column_sum = np.array(features.sum(0))
        r_inv = np.power(column_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features = features.dot(r_mat_inv)

        dist = distance.pdist(features, metric='euclidean')
        dist = distance.squareform(dist)
        sigma = np.mean(dist)
        w = np.exp(- dist ** 2 / (2 * sigma ** 2))
        w[w < sparsity_threshold] = 0
        apoe_affinity *= w
        age_affinity *= w
        gender_affinity *= w
        fdg_affinity *= w

        mixed_affinity = (age_affinity + gender_affinity + fdg_affinity + apoe_affinity) / 4

        c_1 = [i for i in range(num_nodes) if labels[i] == 0]
        c_2 = [i for i in range(num_nodes) if labels[i] == 1]
        c_3 = [i for i in range(num_nodes) if labels[i] == 2]

        # imbalanced
        c_1_num = len(c_1)
        c_2_num = len(c_2)
        c_3_num = len(c_3)
        num_nodes = c_1_num + c_2_num + c_3_num
        np.random.shuffle(c_1)
        np.random.shuffle(c_2)
        np.random.shuffle(c_3)
        selection_c_1 = c_1[:c_1_num]
        selection_c_2 = c_2[:c_2_num]
        selection_c_3 = c_3[:c_3_num]
        idx = np.concatenate((selection_c_1, selection_c_2, selection_c_3), axis=0)
        node_weights = np.zeros((num_nodes,))
        node_weights[selection_c_1] = 1 - c_1_num / float(num_nodes)
        node_weights[selection_c_2] = 1 - c_2_num / float(num_nodes)
        node_weights[selection_c_3] = 1 - c_3_num / float(num_nodes)
        np.random.shuffle(idx)
        features = features[idx, :]
        labels = [labels[item] for item in idx]

        age_affinity = age_affinity[idx, :]
        age_affinity = age_affinity[:, idx]

        gender_affinity = gender_affinity[idx, :]
        gender_affinity = gender_affinity[:, idx]

        fdg_affinity = fdg_affinity[idx, :]
        fdg_affinity = fdg_affinity[:, idx]

        apoe_affinity = apoe_affinity[idx, :]
        apoe_affinity = apoe_affinity[:, idx]

        # adj = adj[idx, :]
        # adj = adj[:, idx]
        node_weights = node_weights[idx]

        print(idx)
        # plot features
        # pca = PCA(n_components=5)
        # pca.fit_transform(features)
        # transformed = pca.transform(features)
        # print(pca.explained_variance_)
        # print(pca.components_)
        # print(pca.mean_)
        # components = [0, 3]
        # plt.scatter(transformed[:, components[0]], transformed[:, components[1]], c=labels,
        #             cmap=plt.cm.get_cmap('spectral', 3))
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')
        # plt.colorbar()
        # plt.show()

        # train_proportion = 0.8
        # val_proportion = 0.1

        # train_mask = np.zeros((num_nodes,), dtype=np.bool)
        # val_mask = np.zeros((num_nodes,), dtype=np.bool)
        # test_mask = np.zeros((num_nodes,), dtype=np.bool)
        # train_mask[:int(train_proportion * num_nodes)] = 1
        # val_mask[int(train_proportion * num_nodes): int((train_proportion + val_proportion) * num_nodes)] = 1
        # test_mask[int((train_proportion + val_proportion) * num_nodes):] = 1

        num_labels = 3
        one_hot_labels = np.zeros((num_nodes, num_labels))
        one_hot_labels[np.arange(num_nodes), labels] = 1

        # train_label = np.zeros(one_hot_labels.shape)
        # val_label = np.zeros(one_hot_labels.shape)
        # test_label = np.zeros(one_hot_labels.shape)
        # train_label[train_mask, :] = one_hot_labels[train_mask,:]
        # val_label[val_mask, :] = one_hot_labels[val_mask, :]
        # test_label[test_mask, :] = one_hot_labels[test_mask, :]

        # train_mask = node_weights * train_mask
        # val_mask = node_weights * val_mask
        # test_mask = node_weights * test_mask
        # SVM performance
        # train_features = features[train_idx, :]
        # train_labels = [labels[i] for i in train_idx]
        # test_features = features[test_idx, :]
        # test_labels = [labels[i] for i in test_idx]
        # svc2 = svm.SVC(kernel='linear').fit(train_features, train_labels)
        # train_pred = svc2.predict(train_features)
        # test_pred = svc2.predict(test_features)
        # print('test acc:', np.mean(np.equal(test_pred, test_labels)))
        # print('train acc:', np.mean(np.equal(train_pred, train_labels)))
        sparse_features = sparse_to_tuple(sp.coo_matrix(features))

        # return adj, features, train_label, val_label, test_label, train_mask, val_mask, test_mask, labels
        return age_affinity, gender_affinity, fdg_affinity, apoe_affinity, mixed_affinity, sparse_features, labels, one_hot_labels, node_weights, features

def main():
    sparsity_threshold = 0.85
    age_adj, gender_adj, fdg_adj, apoe_adj, mixed_adj, features, all_labels, one_hot_labels, node_weights, dense_features = \
        load_tadpole_data(sparsity_threshold)

    tmp_adj = apoe_adj
    new_edges = []
    for i in range(tmp_adj.shape[0]):
        for j in range(i+1, tmp_adj.shape[0]):
            if tmp_adj[i, j] > 0:
                new_edges.append([i, j])
    new_edges = np.array(new_edges)
    np.savez('./data/tadpole/data.npz', ft=dense_features, lbl=all_labels, edge=new_edges)


if __name__ == '__main__':
    main()
