import numpy as np
import scipy.sparse as sp
import data.abide.ABIDEParser as Reader
from sklearn.model_selection import StratifiedKFold


def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class dataloader():
    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = 200
        self.num_classes = 2

    def load_data(self, connectivity='correlation', atlas='ho'):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''
        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)

        sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        genders = Reader.get_subject_score(subject_IDs, score='SEX')

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]]) - 1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]

        self.y = y - 1

        self.raw_features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:, 0] = site
        phonetic_data[:, 1] = gender
        phonetic_data[:, 2] = age

        self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['SEX'] = np.copy(phonetic_data[:, 1])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:, 2])

        return self.raw_features, self.y, phonetic_data

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_node_features(self, train_ind):
        '''preprocess node features for ev-gcn
        '''
        node_ftr = Reader.feature_selection(self.raw_features, self.y, train_ind, self.node_ftr_dim)
        self.node_ftr = preprocess_features(node_ftr)
        return self.node_ftr

    def get_PAE_inputs(self, nonimg):
        '''get PAE inputs for ev-gcn
        '''
        # construct edge network inputs
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # static affinity score used to pre-prune edges
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr, self.pd_dict)
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 1.5)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input


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


def load_ABIDE_data():
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')

    # Get acquisition site
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    num_classes = 2
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=np.int)

    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    features = Reader.get_networks(subject_IDs, kind='correlation', atlas_name='ho')
    #gender_adj = np.zeros((num_nodes, num_nodes))
    gender_adj = Reader.create_affinity_graph_from_scores(['SEX'], subject_IDs)
    #site_adj = np.zeros((num_nodes, num_nodes))
    site_adj = Reader.create_affinity_graph_from_scores([ 'SITE_ID'], subject_IDs)
    mixed_adj = gender_adj+ site_adj

    c_1 = [i for i in range(num_nodes) if y[i] == 1]
    c_2 = [i for i in range(num_nodes) if y[i] == 2]

    # print(idx)
    y_data = np.asarray(y_data, dtype=int)
    num_labels = 2
    #one_hot_labels = np.zeros((num_nodes, num_labels))
    #one_hot_labels[np.arange(num_nodes), y_data] = 1
    sparse_features = sparse_to_tuple(sp.coo_matrix(features))

    features_modi = Reader.feature_selection(features, y, np.arange(0,y.shape[0]).tolist(), 2000)
    return gender_adj, site_adj, mixed_adj, sparse_features, y ,y_data, features_modi


def main():
    # gender_adj, site_adj, mixed_adj, features, all_labels, one_hot_labels, dense_features \
    #     = load_ABIDE_data()

    # tmp_adj = mixed_adj
    # new_edges = []
    # for i in range(tmp_adj.shape[0]):
    #     for j in range(i+1, tmp_adj.shape[0]):
    #         if tmp_adj[i, j] > 0:
    #             new_edges.append([i, j])
    # all_labels = all_labels - 1
    # all_labels = np.asarray(all_labels, dtype=int)
    # all_labels = all_labels.squeeze(1)
    # new_edges = np.array(new_edges)

    dl = dataloader()
    raw_features, y, nonimg = dl.load_data()
    train_ind = np.arange(0, y.shape[0])
    node_ftr = dl.get_node_features(train_ind)
    edge_index, edgenet_input = dl.get_PAE_inputs(nonimg)
    y.dtype = 'int'
    new_edges = edge_index.T.copy()
    print(new_edges.shape)

    np.savez('./data/abide/data.npz', ft=node_ftr, lbl=y, edge=new_edges)


if __name__ == '__main__':
    main()
