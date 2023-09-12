import numpy as np
import torch
import dhg


def main():
    data = dhg.data.Cora()
    ft = data['features'].numpy()
    lbl = data['labels'].numpy()
    g = dhg.Graph(data['num_vertices'], data['edge_list'])
    edge = np.array(g.e[0])
    train_mask = data['train_mask'].numpy()
    val_mask = data['val_mask'].numpy()
    test_mask = data['test_mask'].numpy()
    np.savez('./data/cora/data.npz', ft=ft, lbl=lbl, edge=edge,
             train_mask=train_mask, val_mask = val_mask, test_mask = test_mask)

    # for i in range(np.max(lbl) + 1):
    #     print(f'{i} {np.sum(lbl==i)}')
    # deg = np.zeros(lbl.shape[0])
    # for i in range(edge.shape[0]):
    #     deg[edge[i,0]] += 1
    #     deg[edge[i,1]] += 1
    # cnt = np.zeros(200)
    # for i in range(lbl.shape[0]):
    #     cnt[int(deg[i])] += 1
    # for i in range(1, 51):
    #     print(int(cnt[i]))


if __name__ == '__main__':
    main()