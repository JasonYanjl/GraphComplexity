import numpy as np
import torch
import dhg


def main():
    data = dhg.data.Cooking200()
    ft = np.eye(data['num_vertices'])
    lbl = data['labels'].numpy()
    hg = dhg.Hypergraph(data['num_vertices'], data['edge_list'])
    g = dhg.Graph.from_hypergraph_clique(hg)
    edge = np.array(g.e[0])
    train_mask = data['train_mask'].numpy()
    val_mask = data['val_mask'].numpy()
    test_mask = data['test_mask'].numpy()
    np.savez('./data/cooking/data.npz', ft=ft, lbl=lbl, edge=edge,
             train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)


if __name__ == '__main__':
    main()