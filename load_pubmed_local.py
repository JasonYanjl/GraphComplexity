import numpy as np
from torch_geometric.datasets import Planetoid
import os
import dhg

def main():
    path = "./data/planetoid_data/PubMed/"
    dataset = "PubMed"
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abs_path, path)
    data = Planetoid(root=path, name=dataset)
    data = data[0]
    local_max = 8000
    features = data.x.numpy()
    features = features[:local_max, :]
    labels = data.y.numpy()
    labels = labels[:local_max]
    edges = data.edge_index.T.numpy()
    new_edges = []
    for i in range(edges.shape[0]):
        if edges[i, 0] < local_max and edges[i, 1] < local_max:
            new_edges.append([edges[i, 0], edges[i, 1]])
    new_edges = np.array(new_edges)

    new_data = dhg.data.Pubmed()
    train_mask = new_data['train_mask'].numpy()
    val_mask = new_data['val_mask'].numpy()
    test_mask = new_data['test_mask'].numpy()
    np.savez('./data/planetoid_data/PubMed/data.npz', ft=features, lbl=labels, edge=new_edges,
             train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)


if __name__ == "__main__":
    main()
