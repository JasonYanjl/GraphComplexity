import numpy as np
import torch
import dhg


def main():
    data = dhg.data.Citeseer()
    ft = data['features'].numpy()
    lbl = data['labels'].numpy()
    g = dhg.Graph(data['num_vertices'], data['edge_list'])
    edge = np.array(g.e[0])
    train_mask = data['train_mask'].numpy()
    val_mask = data['val_mask'].numpy()
    test_mask = data['test_mask'].numpy()
    np.savez('./data/citeseer/data.npz', ft=ft, lbl=lbl, edge=edge,
             train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)


if __name__ == '__main__':
    main()