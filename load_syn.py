import numpy as np
import os
import dhg
import matplotlib.pyplot as plt
import random


def main():
    N = 300
    E = 800
    h = 0.9

    cls = 3

    A = np.zeros((N, N))

    lbl = np.zeros(N, dtype=int)
    deg = np.zeros(N, dtype=int)
    for i in range(N):
        lbl[i] = i // (N / cls)
    np.random.shuffle(lbl)

    edge = np.zeros((E, 2), dtype=int)

    tote = 0
    for i in range(0, N):
        now_p = np.zeros(N)
        if deg[i] > 0:
            continue

        if np.random.random() <= h:  # in-cluster
            for j in range(0, N):
                if i == j:
                    continue
                if lbl[i] != lbl[j]:
                    continue
                if A[i, j] > 0:
                    continue
                now_p[j] = max(deg[j], 1)

        else:  # out-cluster
            for j in range(0, N):
                if i == j:
                    continue
                if lbl[i] == lbl[j]:
                    continue
                if A[i, j] > 0:
                    continue
                now_p[j] = max(deg[j], 1)

        now_p = now_p / np.sum(now_p)
        now_j = np.random.choice(np.arange(0, N), size=1, p=now_p)[0]
        edge[tote, 0] = i
        edge[tote, 1] = now_j
        deg[i] += 1
        deg[now_j] += 1
        A[i, now_j] = 1
        A[now_j, i] = 1
        tote += 1

    while tote < E:
        i = np.random.randint(0, N)
        now_p = np.zeros(N)

        if np.random.random() <= h:  # in-cluster
            for j in range(0, N):
                if i == j:
                    continue
                if lbl[i] != lbl[j]:
                    continue
                if A[i, j] > 0:
                    continue
                now_p[j] = max(deg[j], 1)


        else:  # out-cluster
            for j in range(0, N):
                if i == j:
                    continue
                if lbl[i] == lbl[j]:
                    continue
                if A[i, j] > 0:
                    continue
                now_p[j] = max(deg[j], 1)

        now_p = now_p / np.sum(now_p)
        now_j = np.random.choice(np.arange(0, N), size=1, p=now_p)[0]
        edge[tote, 0] = i
        edge[tote, 1] = now_j
        deg[i] += 1
        deg[now_j] += 1
        A[i, now_j] = 1
        A[now_j, i] = 1
        tote += 1

    # cnt = np.zeros(200)
    # for i in range(lbl.shape[0]):
    #     cnt[int(deg[i])] += 1
    # for i in range(1, 51):
    #     print(int(cnt[i]))

    real_data = dhg.data.Cora()
    real_ft = real_data['features'].numpy()
    real_lbl = real_data['labels'].numpy()

    ft = np.zeros((N, real_ft.shape[1]))
    # ft = real_ft

    class_map = [0, 2, 3, 1, 4, 5, 6]

    for i in range(cls):
        loc = np.where(lbl == i)[0]
        real_loc = np.where(real_lbl == class_map[i])[0]
        np.random.shuffle(real_loc)
        tmp_ft = real_ft[real_loc[:loc.shape[0]], :]
        ft[loc, :] = tmp_ft

    np.savez('./data/syn-cora/data.npz', ft=ft, lbl=lbl, edge=edge)

    # graph2wolfram_str(lbl, edge)
    draw(lbl, edge, cls)


def graph2wolfram_str(lbl, edge):
    # print(lbl)
    # print(edge)
    res_str = ''
    res_str += 'Graph[{'
    for i in range(lbl.shape[0]):
        res_str += f'Style[{i+1},'
        if lbl[i] == 0:
            res_str += 'Red]'
        elif lbl[i] == 1:
            res_str += 'Blue]'
        elif lbl[i] == 2:
            res_str += 'Green]'
        if i+1 == lbl.shape[0]:
            res_str += '},'
        else:
            res_str += ','
    res_str += '{'
    for i in range(edge.shape[0]):
        res_str += f'{edge[i, 0]+1} <-> {edge[i, 1]+1}'
        if i + 1 == edge.shape[0]:
            res_str += '}'
        else:
            res_str += ','
    res_str += ']'
    print(res_str)


def draw(lbl, edge, num_classes):
    variance_factor = 40
    start_cov = np.array(
        [[70.0 * variance_factor, 0.0],
         [0.0, 20.0 * variance_factor]])
    cov = start_cov
    theta = np.pi * 2 / num_classes
    rotation_mat = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    radius = 300
    allx = np.zeros(shape=[lbl.shape[0], 2], dtype='float32')
    plt.figure(figsize=(40, 40))
    for cls, theta in enumerate(np.arange(0, np.pi * 2, np.pi * 2 / num_classes)):
        gaussian_y = radius * np.cos(theta)
        gaussian_x = radius * np.sin(theta)
        num_points = np.sum(lbl == cls)
        coord_x, coord_y = np.random.multivariate_normal(
            [gaussian_x, gaussian_y], cov, num_points).T
        cov = rotation_mat.T.dot(cov.dot(rotation_mat))

        # Belonging to class cls
        example_indices = np.nonzero(lbl==cls)[0]
        random.shuffle(example_indices)
        allx[example_indices, 0] = coord_x
        allx[example_indices, 1] = coord_y

    num_edges = len(edge)
    permutation = np.random.permutation(num_edges)
    random.shuffle(permutation)
    for v1, v2 in [edge[i] for i in permutation[:]]:
        xx = [allx[v1][0], allx[v2][0]]
        yy = [allx[v1][1], allx[v2][1]]
        plt.plot(xx, yy, color='silver', linestyle='-', linewidth=1)
    for cls, theta in enumerate(np.arange(0, np.pi * 2, np.pi * 2 / num_classes)):
        example_indices = np.nonzero(lbl==cls)[0]
        plt.plot(allx[example_indices, 0], allx[example_indices, 1], 'o', markersize=20)
    plt.savefig(f'tmp{edge.shape[0]}.jpg')


if __name__ == '__main__':
    main()
