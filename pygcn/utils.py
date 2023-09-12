import numpy as np
import scipy.sparse as sp
import torch
import os
# from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import json


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def save_graph2json(n, edges, labels):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abs_path, '../plot/graph_json.json')
    color_list = ['#3C76AF', '#EE8636', '#519D40', '#C33934', '#9566BC', '#84584D', '#D67DBF']
    res_list = []
    _labels = np.where(labels)[1]
    name_list = []
    for i in range(n):
        name_list.append(f'flare.type{_labels[i]}.node{i}')
    link_list = [[] for i in range(n)]
    for i in range(edges.shape[0]):
        link_list[int(edges[i, 0])].append(name_list[int(edges[i, 1])])
    for i in range(n):
        res_list.append({
            'name': name_list[i],
            "imports": link_list[i],
            "label": int(_labels[i]),
            "color": color_list[_labels[i]]
        })
    res_list.sort(key=lambda x:x['label'])
    with open(path, "w") as f:
        json.dump(res_list, f)


def save_graph2csv(n, edges, labels, prev_mat=None):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abs_path, '../plot/graph_csv.csv')
    f = open(path, 'w', encoding='gbk')

    _labels = np.where(labels)[1]
    tot_label = np.max(_labels) + 1
    acc_e = 0
    for i in range(edges.shape[0]):
        if _labels[int(edges[i, 0])] == _labels[int(edges[i, 1])]:
            acc_e += 1
    tmp_h = acc_e / edges.shape[0]
    print(tmp_h)

    sum_n = np.zeros(tot_label)
    for i in range(tot_label):
        sum_n[i] = np.sum(_labels == i)

    edge_matrix = np.zeros((tot_label, tot_label))

    for i in range(tot_label):
        for j in range(edges.shape[0]):
            if _labels[int(edges[j, 0])] == i:
                edge_matrix[i, int(_labels[int(edges[j, 1])])] += 1
            if _labels[int(edges[j, 1])] == i:
                edge_matrix[i, int(_labels[int(edges[j, 0])])] += 1
        if np.sum(edge_matrix[i, :]) == 0:
            for j in range(tot_label):
                if i != j:
                    if prev_mat is None:
                        edge_matrix[i, j] = sum_n[i] * (1 - tmp_h) * sum_n[j] / (np.sum(sum_n) - sum_n[i])
                    else:
                        edge_matrix[i, j] = sum_n[i] * (1 - tmp_h) * prev_mat[i, j] / (np.sum(prev_mat[i, :]) - prev_mat[i, i])
                else:
                    edge_matrix[i, j] = sum_n[i] * tmp_h
        else:
            edge_matrix[i, :] = edge_matrix[i, :] / np.sum(edge_matrix[i,: ]) * sum_n[i]

    for i in range(tot_label):
        for j in range(tot_label):
            f.write(f"{edge_matrix[i, j]}")
            if j != tot_label - 1:
                f.write(",")
            else:
                f.write("\n")
    f.close()

    return edge_matrix


def js_div(x, y):
    med = (x + y) / 2
    res = 0
    for i in range(x.shape[0]):
        if x[i] > 0:
            res += 1 / 2 * x[i] * np.log(x[i] / med[i])
        if y[i] > 0:
            res += 1 / 2 * y[i] * np.log(y[i] / med[i])
    return res


def calc_complexity_fast(n, edges):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = np.zeros((n, n))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
    d = np.array(np.sum(A, axis=0)).reshape(-1)

    A_torch = torch.tensor(A).to(device)
    pp11 = (torch.matmul(A_torch, A_torch) / (n - 2)).cpu().numpy()
    torch.cuda.empty_cache()
    pp01 = (torch.matmul((1 - A_torch), A_torch) / (n - 2)).cpu().numpy()
    torch.cuda.empty_cache()
    pp10 = (torch.matmul(A_torch, (1 - A_torch)) / (n - 2)).cpu().numpy()
    torch.cuda.empty_cache()
    A_torch_1 = 1 - A_torch
    A_torch = None
    torch.cuda.empty_cache()
    pp00 = (torch.matmul(A_torch_1, A_torch_1) / (n - 2)).cpu().numpy()
    A_torch_1 = None
    torch.cuda.empty_cache()

    p1 = d / (n - 1)
    p0 = 1 - p1
    K = np.zeros(n)
    for i in range(n):
        if p1[i] > 0:
            K[i] -= p1[i] * np.log(p1[i]) / np.log(2)
        if p0[i] > 0:
            K[i] -= p0[i] * np.log(p0[i]) / np.log(2)

    m = np.zeros((n, n))

    tmp = pp00 * np.log(pp00 / np.tile(p0, (n, 1)).T / np.tile(p0, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    tmp = pp01 * np.log(pp01 / np.tile(p0, (n, 1)).T / np.tile(p1, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    tmp = pp10 * np.log(pp10 / np.tile(p1, (n, 1)).T / np.tile(p0, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    tmp = pp11 * np.log(pp11 / np.tile(p1, (n, 1)).T / np.tile(p1, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    res = np.sum(m * (1 - m) * np.maximum(np.tile(K, (n, 1)).T, np.tile(K, (n, 1))))
    res = res / (n * (n - 1))

    print(f'complexity: {res}')

    return res


def calc_complexity(n, edges):
    A = np.mat(np.zeros((n, n)))
    d = np.zeros(n)
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
        d[edges[i][0]] += 1
        d[edges[i][1]] += 1

    pp = np.zeros((2, 2, n, n))
    pp[0, 0, :, :] = np.matmul((1 - A), (1 - A)) / (n - 2)
    pp[1, 1, :, :] = np.matmul(A, A) / (n - 2)
    pp[0, 1, :, :] = np.matmul((1 - A), A) / (n - 2)
    pp[1, 0, :, :] = np.matmul(A, (1 - A)) / (n - 2)
    p = np.zeros((2, n))
    p[1, :] = d / (n - 1)
    p[0, :] = 1 - p[1, :]
    K = np.zeros(n)
    for i in range(n):
        if p[1, i] > 0:
            K[i] -= p[1, i] * np.log(p[1, i]) / np.log(2)
        if p[0, i] > 0:
            K[i] -= p[0, i] * np.log(p[0, i]) / np.log(2)
    m = np.zeros((n, n))
    res = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for a in range(2):
                for b in range(2):
                    if pp[a, b, i, j] > 0:
                        m[i, j] += pp[a, b, i, j] * np.log(pp[a, b, i, j] / p[a, i] / p[b, j]) / np.log(2)
            res = res + max(K[i], K[j]) * m[i, j] * (1 - m[i, j])
    res = res / (n * (n - 1))
    print(f'complexity: {res}')
    return res


def calc_dis_mutual_info(n, edges):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = np.zeros((n, n))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
    d = np.array(np.sum(A, axis=0)).reshape(-1)

    A_torch = torch.tensor(A).to(device)
    pp11 = (torch.matmul(A_torch, A_torch) / (n - 2)).cpu().numpy()
    torch.cuda.empty_cache()
    pp01 = (torch.matmul((1 - A_torch), A_torch) / (n - 2)).cpu().numpy()
    torch.cuda.empty_cache()
    pp10 = (torch.matmul(A_torch, (1 - A_torch)) / (n - 2)).cpu().numpy()
    torch.cuda.empty_cache()
    A_torch_1 = 1 - A_torch
    A_torch = None
    torch.cuda.empty_cache()
    pp00 = (torch.matmul(A_torch_1, A_torch_1) / (n - 2)).cpu().numpy()
    A_torch_1 = None
    torch.cuda.empty_cache()

    p1 = d / (n - 1)
    p0 = 1 - p1
    K = np.zeros(n)
    for i in range(n):
        if p1[i] > 0:
            K[i] -= p1[i] * np.log(p1[i]) / np.log(2)
        if p0[i] > 0:
            K[i] -= p0[i] * np.log(p0[i]) / np.log(2)

    m = np.zeros((n, n))

    tmp = pp00 * np.log(pp00 / np.tile(p0, (n, 1)).T / np.tile(p0, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    tmp = pp01 * np.log(pp01 / np.tile(p0, (n, 1)).T / np.tile(p1, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    tmp = pp10 * np.log(pp10 / np.tile(p1, (n, 1)).T / np.tile(p0, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    tmp = pp11 * np.log(pp11 / np.tile(p1, (n, 1)).T / np.tile(p1, (n, 1))) / np.log(2)
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    m += tmp

    dis = np.zeros((n, n))
    tmpA = np.copy(A)
    for i in range(1, 7):
        res = []
        x = np.where(A > 0)[0]
        y = np.where(A > 0)[1]
        for j in range(x.shape[0]):
            if dis[x[j], y[j]] == 0:
                dis[x[j], y[j]] = i
                if x[j] != y[j]:
                    res.append(m[x[j], y[j]])
        A = np.matmul(A, tmpA)
        A[A > 0] = 1
        print(f'dis {i} avg mutual info: {np.mean(np.array(res))}')

    return m, dis


def calc_fts_entropy(edges, ft):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft = ft.clone().detach().to(device)
    ft_dot = torch.matmul(ft, ft.T)
    sele = ft_dot[edges[:, 0], edges[:, 1]]
    sele = torch.softmax(sele, dim=0)
    res = -torch.sum(torch.log(sele) * sele).item()
    print(f'fts_entropy: {res}')
    print(res / edges.shape[0])
    return res


def calc_ptoq_edge_trainval(lbl, edges, mutual_info, dis, train_mask, val_mask):
    n = lbl.shape[0]
    A = np.mat(np.zeros((n, n)))
    tote = 0
    acce = 0
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1

        if (train_mask[edges[i][0]] or val_mask[edges[i][0]]) \
                and (train_mask[edges[i][1]] or val_mask[edges[i][1]]):
            tote += 1
            if lbl[edges[i][0]] == lbl[edges[i][1]]:
                acce += 1

    std_g = np.mat(np.zeros((n, n)))
    for i in range(n):
        for j in range(i + 1, n):
            if lbl[i] == lbl[j]:
                std_g[i, j] = 1
                std_g[j, i] = 1

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                nedges.append([i, j, mutual_info[i, j]])
    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    Ran = int(edges.shape[0] * 0.04)
    for i in range(0, edges.shape[0] * 5, Ran):
        range_res = []
        for j in range(i, i + Ran, 1):
            if (train_mask[int(nedges[j, 0])] or val_mask[int(nedges[j, 0])]) \
                    and (train_mask[int(nedges[j, 1])] or val_mask[int(nedges[j, 1])]):
                tote += 1
                if std_g[int(nedges[j, 0]), int(nedges[j, 1])] == 0:
                    pass
                else:
                    acce += 1
            range_res.append(acce / (tote - acce))
        print(np.mean(np.array(range_res)))

    print('-' * 50)

    Ran = int(edges.shape[0] * 0.04)
    range_acc = 0
    range_tot = 0
    for i in range(0, edges.shape[0] * 5, Ran):
        for j in range(i, i + Ran, 1):
            if (train_mask[int(nedges[j, 0])] or val_mask[int(nedges[j, 0])]) \
                    and (train_mask[int(nedges[j, 1])] or val_mask[int(nedges[j, 1])]):
                range_tot += 1
                if std_g[int(nedges[j, 0]), int(nedges[j, 1])] == 0:
                    pass
                else:
                    range_acc += 1
        if range_tot == range_acc:
            print(0)
        else:
            print(range_acc / (range_tot - range_acc))


def calc_ptoq_edge(lbl, edges, mutual_info, dis):
    n = lbl.shape[0]
    A = np.mat(np.zeros((n, n)))
    tote = 0
    acce = 0
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
        tote += 1
        if lbl[edges[i][0]] == lbl[edges[i][1]]:
            acce += 1

    std_g = np.mat(np.zeros((n, n)))
    for i in range(n):
        for j in range(i + 1, n):
            if lbl[i] == lbl[j]:
                std_g[i, j] = 1
                std_g[j, i] = 1

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                nedges.append([i, j, mutual_info[i, j]])
    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    Ran = int(edges.shape[0] * 0.04)
    for i in range(0, edges.shape[0] * 5, Ran):
        range_res = []
        for j in range(i, i + Ran, 1):
            tote += 1
            if std_g[int(nedges[j, 0]), int(nedges[j, 1])] == 0:
                pass
            else:
                acce += 1
            range_res.append(acce / tote)
        print(np.mean(np.array(range_res)))

    print('-' * 50)

    Ran = int(edges.shape[0] * 0.04)
    range_acc = 0
    range_tot = 0
    for i in range(0, edges.shape[0] * 5, Ran):
        for j in range(i, i + Ran, 1):
            range_tot += 1
            if std_g[int(nedges[j, 0]), int(nedges[j, 1])] == 0:
                pass
            else:
                range_acc += 1

        print(range_acc / range_tot)


def calc_homophily(lbl, edges):
    n = lbl.shape[0]
    tote = 0
    acce = 0
    for i in range(edges.shape[0]):
        # print(i,' ', edges[i][0],' ',edges[i][1], ' ', lbl[edges[i][0]], ' ', lbl[edges[i][1]])
        tote += 1
        if lbl[edges[i][0]] == lbl[edges[i][1]]:
            acce += 1
    print(f'homophily: {acce / tote}')
    return acce / tote



def calc_ptoq_edge_delete(lbl, edges, mutual_info, dis):
    n = lbl.shape[0]
    A = np.mat(np.zeros((n, n)))
    tote = 0
    acce = 0
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
        tote += 1
        if lbl[edges[i][0]] == lbl[edges[i][1]]:
            acce += 1

    std_g = np.mat(np.zeros((n, n)))
    for i in range(n):
        for j in range(i + 1, n):
            if lbl[i] == lbl[j]:
                std_g[i, j] = 1
                std_g[j, i] = 1

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                nedges.append([i, j, mutual_info[i, j]])
    nedges = np.array(nedges)
    nedges = nedges[np.argsort(nedges[:, 2])]

    print('剩余边p/q')
    print(acce / (tote))
    Ran = int(edges.shape[0] * 0.02)
    for i in range(0, edges.shape[0], Ran):
        range_res = []
        for j in range(i, min(i + Ran, edges.shape[0]), 1):
            tote -= 1
            if std_g[int(nedges[j, 0]), int(nedges[j, 1])] == 0:
                pass
            else:
                acce -= 1
            if tote == acce:
                range_res.append(0)
            else:
                range_res.append(acce / (tote))
        print(i,' ',np.mean(np.array(range_res)))

    print('-' * 50)

    print('删除边p/q')
    Ran = int(edges.shape[0] * 0.02)
    range_acc = 0
    range_tot = 0
    for i in range(0, edges.shape[0], Ran):
        for j in range(i, min(i + Ran, edges.shape[0]), 1):
            range_tot += 1
            if std_g[int(nedges[j, 0]), int(nedges[j, 1])] == 0:
                pass
            else:
                range_acc += 1

        if range_tot == range_acc:
            print(0)
        else:
            print(range_acc / (range_tot - range_acc))


def calc_delta_edge(lbl, edges, mutual_info, dis):
    n = lbl.shape[0]
    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1

    std_g = np.mat(np.zeros((n, n)))
    for i in range(n):
        for j in range(i + 1, n):
            if lbl[i] == lbl[j]:
                std_g[i, j] = 1
                std_g[j, i] = 1

    # a_diff = 0
    # b_diff = 0
    diff = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != std_g[i, j]:
                diff += 1
                # if A[i, j] == 1:
                #     a_diff+=1
                # else:
                #     b_diff+=1

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                nedges.append([i, j, mutual_info[i, j]])
    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    Ran = int(edges.shape[0] * 0.1)
    for i in range(0, int(edges.shape[0] * 12.5), Ran):
        range_res = []
        for j in range(i, i + Ran, 1):
            if std_g[int(nedges[j, 0]), int(nedges[j, 1])] == 0:
                diff += 1
            else:
                diff -= 1
            range_res.append(diff)
        print(np.mean(np.array(range_res)))


def calc_fts_mutual(n, edges, mutual_info, dis, features):
    ft_2 = np.array(torch.matmul(features, features.T).cpu().detach())
    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                nedges.append([i, j, mutual_info[i, j], ft_2[i, j]])
    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    Ran = int(edges.shape[0] * 0.1)
    for i in range(0, edges.shape[0] * 15, Ran):
        print(np.mean(nedges[i:i + Ran, 3]))


def select_2_ratio(edges, lbl, ratio):
    p_edges = []
    q_edges = []
    tmplbl = np.where(lbl)[1]
    for i in range(edges.shape[0]):
        if tmplbl[edges[i, 0]] == tmplbl[edges[i, 1]]:
            p_edges.append([int(edges[i, 0]), int(edges[i, 1])])
        else:
            q_edges.append([int(edges[i, 0]), int(edges[i, 1])])

    p_edges = np.array(p_edges)
    q_edges = np.array(q_edges)
    q_siz = int(p_edges.shape[0] / ratio)
    res = None
    if q_siz > q_edges.shape[0]:
        p_siz = int(q_edges.shape[0] * ratio)
        if p_siz > p_edges.shape[0]:
            print(f'ratio set error')
            exit(0)
        else:
            np.random.shuffle(p_edges)
            res = np.concatenate((p_edges[:p_siz, :], q_edges))
    else:
        np.random.shuffle(q_edges)
        res = np.concatenate((p_edges, q_edges[:q_siz, :]))
    np.random.shuffle(res)
    return res


def calc_lbl_mutual(lbl, edges, mutual_info, dis, features):
    n = lbl.shape[0]
    lb_2 = np.zeros((n, torch.max(lbl) + 1))
    for i in range(n):
        lb_2[i, lbl[i]] += 1

    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
        lb_2[edges[i][0], lbl[edges[i][1]]] += 1
        lb_2[edges[i][1], lbl[edges[i][0]]] += 1

    lb_2 = (lb_2.T / np.sum(lb_2.T, axis=0)).T

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                nedges.append([i, j, mutual_info[i, j]])
    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    Ran = int(edges.shape[0] * 0.1)
    for i in range(0, edges.shape[0] * 15, Ran):
        tmp_res = []
        for j in range(i, i + Ran):
            tmp_res.append(js_div(lb_2[int(nedges[j, 0]), :], lb_2[int(nedges[j, 1]), :]))
        print(np.mean(np.array(tmp_res)))


def add_noise(n, edges, cnt):
    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1

    np.random.shuffle(edges)
    nedges = []
    for i in range(cnt):
        chk = False
        while not chk:
            x = np.random.randint(0, n)
            y = np.random.randint(0, n)
            if x == y or A[x, y] == 1 or A[y, x] == 1:
                chk = False
            else:
                chk = True
                A[x, y] = 1
                A[y, x] = 1
                nedges.append([x, y])
    edges[:cnt, :] = nedges
    return edges


def addedges(n, edges, cnt):
    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
    B = np.matmul(A, A)
    B[B > 0] = 1
    B = np.logical_or(A, B) - A - np.eye(n)
    x = np.where(B > 0)[0]
    y = np.where(B > 0)[1]
    x = np.expand_dims(x, 0)
    y = np.expand_dims(y, 0)
    nedges = np.concatenate((x, y), axis=0).T
    idx = np.random.choice(nedges.shape[0], cnt, replace=False)
    return np.concatenate((edges, nedges[idx, :]))


def addedges_d(n, edges, cnt):
    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1
    d = np.array(np.sum(A, axis=0)).reshape(-1)
    mxw = np.argpartition(d, 100)[:100]
    mnw = np.argpartition(-d, 100)[:100]
    nedges = np.zeros((0, 2))
    for i in range(mxw.shape[0]):
        for j in range(i + 1, mxw.shape[0]):
            nedges = np.concatenate((nedges, np.array([[mxw[i], mxw[j]]])))
    nedges = nedges.astype(np.int32)
    idx = np.random.choice(nedges.shape[0], cnt, replace=False)
    return np.concatenate((edges, nedges[idx, :]))


def add_and_del_edges_mutual(n, edges, mutual_info, add_edge_mutual, del_edge_mutual, dis):
    # add
    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                nedges.append([i, j, mutual_info[i, j]])
    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    res_add = nedges[:add_edge_mutual, :2].copy()

    # del
    nedges = []
    for i in range(edges.shape[0]):
        nedges.append([edges[i][0], edges[i][1], mutual_info[edges[i][0], edges[i][1]]])

    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    res_del = nedges[:del_edge_mutual, :2].copy()

    res = np.concatenate((res_add, res_del))
    res = res.astype(np.int64)
    return res


def addedges_mutual(_labels, edges, mutual_info, edge_mutual, dis, param_lambda, known_masks):
    labels = torch.LongTensor(np.where(_labels)[1])
    n = _labels.shape[0]
    A = np.mat(np.zeros((n, n)))

    tote = 0
    acce = 0
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1

        if known_masks[edges[i][0]] and known_masks[edges[i][1]]:
            tote += 1
            if labels[edges[i][0]] == labels[edges[i][1]]:
                acce += 1

    nedges = []
    tot_con = 0
    tot_e_con = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                if known_masks[i] and known_masks[j]:
                    if labels[i] == labels[j]:
                        tmp_v = mutual_info[i, j] + param_lambda * ((acce+1) / (tote+1) - acce / tote)
                        tot_con += ((acce+1) / (tote+1) - acce / tote)
                    else:
                        tmp_v = mutual_info[i, j] + param_lambda * ((acce) / (tote + 1) - acce / tote)
                        tot_con += ((acce) / (tote + 1) - acce / tote)
                    nedges.append([i, j, tmp_v])
                    tot_e_con += 1
    tot_e_con = max(tot_e_con, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and dis[i, j] > 1:
                if known_masks[i] and known_masks[j]:
                    pass
                else:
                    tmp_v = mutual_info[i, j] + param_lambda * tot_con / tot_e_con
                    nedges.append([i, j, tmp_v])

    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    res = np.concatenate((edges, nedges[:edge_mutual, :2]))
    res = res.astype(np.int64)
    return res


def deledges_mutual(n, edges,  mutual_info, edge_mutual, dis):
    nedges = []
    for i in range(edges.shape[0]):
        nedges.append([edges[i][0], edges[i][1], mutual_info[edges[i][0], edges[i][1]]])

    nedges = np.array(nedges)
    nedges = nedges[np.argsort(-nedges[:, 2])]

    res = nedges[:edge_mutual, :2].copy()
    res = res.astype(np.int64)
    return res


def addedges_random(n, edges, edge_mutual):
    A = np.mat(np.zeros((n, n)))
    for i in range(edges.shape[0]):
        A[edges[i][0], edges[i][1]] = 1
        A[edges[i][1], edges[i][0]] = 1

    nedges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                nedges.append([i, j])
    nedges = np.array(nedges)
    np.random.shuffle(nedges)

    return np.concatenate((edges, nedges[:edge_mutual, :2]))


def edge_vis_str(edges):
    res_str = 'Graph[{'
    for i in range(edges.shape[0]):
        res_str = res_str + f'{edges[i][0]}<->{edges[i][1]}'
        if i < edges.shape[0] - 1:
            res_str = res_str + ','
    res_str = res_str + '}]'
    print(res_str)


def show_adj_matrix(edges, labels):
    lbl = np.where(labels)[1]
    mp = np.zeros(lbl.shape[0], dtype=np.int32)
    loc = np.zeros(lbl.shape[0], dtype=np.int32)
    totp = 0
    for k in range(np.max(lbl) + 1):
        for i in range(lbl.shape[0]):
            if lbl[i] == k:
                mp[totp] = i
                loc[i] = totp
                totp += 1

    adj_matrix = np.zeros((lbl.shape[0], lbl.shape[0]))

    is_gt = False
    if is_gt:
        for i in range(lbl.shape[0]):
            for j in range(lbl.shape[0]):
                if lbl[mp[i]] == lbl[mp[j]]:
                    adj_matrix[i, j] = 1
        plt.imshow(adj_matrix, cmap='Blues')
        plt.show()
    else:
        for i in range(lbl.shape[0]):
            adj_matrix[i, i] = 1
        for i in range(edges.shape[0]):
            for i_1 in range(-3, 4, 1):
                for i_2 in range(-3, 4, 1):
                    x_1 = min(max(0, loc[edges[i, 0]] + i_1), lbl.shape[0] - 1)
                    x_2 = min(max(0, loc[edges[i, 1]] + i_2), lbl.shape[0] - 1)
                    adj_matrix[x_1, x_2] = 1
                    adj_matrix[x_2, x_1] = 1
        plt.imshow(adj_matrix, cmap='Blues')
        plt.show()


def load_data(path="../data/BRCA/", dataset="BRCA", rate=1, add_edge_mutual=1, del_edge_mutual=1, output=""):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(abs_path, path)
    noise_rate = 0
    drop_rate = 0

    param_lambda = 20
    # 20

    train_mask = None
    val_mask = None
    test_mask = None

    # build graph
    if dataset == 'cora':
        '''
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        '''
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']
    elif dataset == 'citeseer':
        '''
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        '''
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']
    elif dataset == 'PubMed':
        data = Planetoid(root=path, name=dataset)
        data = data[0]
        features = data.x.numpy()
        features = sp.csr_matrix(features)
        labels = encode_onehot(data.y.numpy())
        edges = data.edge_index.T.numpy()
    elif dataset == 'PubMedSub':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
    elif dataset == 'SFC':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
    elif dataset == 'cooking':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']
    elif dataset == 'acm':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        ori_label = data['lbl'].copy()
        edges = data['edge']
        train_mask = np.full((labels.shape[0]), False)
        train_mask[1000:2000] = True
        val_mask = np.full((labels.shape[0]), False)
        val_mask[2000:3000] = True
    elif dataset == 'Flickr':
        idx_features = np.genfromtxt("{}{}.node".format(path, dataset),
                                     dtype=np.dtype(int))
        features = np.zeros((np.max(idx_features[:, 0]) + 1, np.max(idx_features[:, 1]) + 1))
        for i in range(idx_features.shape[0]):
            features[idx_features[i, 0], idx_features[i, 1]] = 1
        features = sp.csr_matrix(features)
        labels = np.genfromtxt("{}{}.label".format(path, dataset),
                               dtype=np.dtype(int))
        labels = encode_onehot(labels)
        edges = np.genfromtxt("{}{}.edge".format(path, dataset),
                              dtype=np.dtype(int))
        edges = edges[edges[:, 0] < edges[:, 1]]
    elif dataset == 'BlogCatalog':
        idx_features = np.genfromtxt("{}{}.node".format(path, dataset),
                                     dtype=np.dtype(int))
        features = np.zeros((np.max(idx_features[:, 0]) + 1, np.max(idx_features[:, 1]) + 1))
        for i in range(idx_features.shape[0]):
            features[idx_features[i, 0], idx_features[i, 1]] = 1
        features = sp.csr_matrix(features)
        labels = np.genfromtxt("{}{}.label".format(path, dataset),
                               dtype=np.dtype(int))
        labels = encode_onehot(labels)
        edges = np.genfromtxt("{}{}.edge".format(path, dataset),
                              dtype=np.dtype(int))
        edges = edges[edges[:, 0] < edges[:, 1]]
        # train_mask = np.full((labels.shape[0]), False)
        # train_mask[1000:2000] = True
        # val_mask = np.full((labels.shape[0]), False)
        # val_mask[2000:3000] = True
    elif dataset == 'dblp':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
    elif dataset == 'imdb':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
        train_mask = np.full((labels.shape[0]), False)
        train_mask[1500:2000] = True
        val_mask = np.full((labels.shape[0]), False)
        val_mask[2000:3000] = True
        test_mask = np.full((labels.shape[0]), False)
        test_mask[3000:4000] = True
    elif dataset == 'tadpole':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
        # train_mask = np.full((labels.shape[0]), False)
        # train_mask[:200] = True
        # val_mask = np.full((labels.shape[0]), False)
        # val_mask[200:300] = True
        # test_mask = np.full((labels.shape[0]), False)
        # test_mask[300:505] = True
    elif dataset == 'abide':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
    elif dataset == 'syn-cora':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
    elif dataset == 'ROSMAP':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']
    elif dataset == 'BRCA':
        data = np.load(os.path.join(path, 'data.npz'))
        features = data['ft']
        features = sp.csr_matrix(features)
        labels = encode_onehot(data['lbl'])
        edges = data['edge']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']
    else:
        exit(0)

    if drop_rate != 0:
        not_drop_edges = np.random.choice(edges.shape[0], int(edges.shape[0] * (1 - drop_rate)), replace=False)
        edges = edges[not_drop_edges]

    if noise_rate != 0:
        edges = add_noise(labels.shape[0], edges, int(edges.shape[0] * noise_rate))

    prev_csv_matrix = save_graph2csv(labels.shape[0], edges, labels)
    # save_graph2json(labels.shape[0], edges, labels)
    # exit(0)

    ## complexity
    if rate <= 1:
        choice_edges = np.random.choice(edges.shape[0], int(edges.shape[0] * rate), replace=False)
        edges = edges[choice_edges]

    elif rate > 1:
        edges = addedges_d(labels.shape[0], edges, int(edges.shape[0] * (rate - 1)))


    if del_edge_mutual < 1 or add_edge_mutual > 1:
        mutual_info, dis = calc_dis_mutual_info(labels.shape[0], edges)

    if del_edge_mutual < 1:
        # del edges according to mutual information
        edges = deledges_mutual(labels.shape[0], edges, mutual_info, int(edges.shape[0] * del_edge_mutual), dis)

    if add_edge_mutual > 1:
        mutual_info, dis = calc_dis_mutual_info(labels.shape[0], edges)
        edges = addedges_mutual(labels, edges, mutual_info, int(edges.shape[0] * (add_edge_mutual - 1)), dis, param_lambda,
                                np.logical_or(train_mask, val_mask))
        # edges = addedges_mutual(labels, edges, mutual_info, int(edges.shape[0] * (add_edge_mutual - 1)), dis,
        #                         param_lambda,
        #                         np.full((labels.shape[0]), False))
        # edges = addedges_random(labels.shape[0], edges, int(edges.shape[0] * (edge_mutual - 1)))

    # np.savez(output, lbl=ori_label, edge=edges)
    save_graph2csv(labels.shape[0], edges, labels, prev_csv_matrix)
    # save_graph2json(labels.shape[0], edges, labels)
    # exit(0)

    # edges = select_2_ratio(edges, labels, ratio=2.5)

    # np.random.shuffle(edges)
    # edges = edges[:3024]
    calc_complexity_fast(labels.shape[0], edges)
    # calc_complexity(labels.shape[0], edges)

    # if add_edge_mutual > 1 or del_edge_mutual < 1:
    #     edges = add_and_del_edges_mutual(labels.shape[0], edges, mutual_info, int(edges.shape[0] * (add_edge_mutual - 1)),
    #                                      int(edges.shape[0] * del_edge_mutual), dis)

    # edge_vis_str(edges)
    # exit(0)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # feature normalize cannot use in flickr, blogcatalog, ROSMAP, BRCA datasets
    features = normalize(features)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    # idx_train = range(80)
    # idx_val = range(80, 100)
    # idx_test = range(100, 160)

    # show_adj_matrix(edges, labels)
    # exit(0)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)


    # calc_fts_entropy(edges, features)
    # calc_fts_mutual(labels.shape[0], edges, mutual_info, dis, features)
    # calc_lbl_mutual(labels, edges, mutual_info, dis, features)
    # exit(0)
    # calc_delta_edge(labels, edges, mutual_info, dis)
    # exit(0)
    print(labels.shape)
    # print(torch.max(labels))
    print(edges.shape)
    calc_homophily(labels, edges)
    # calc_ptoq_edge(labels, etdges, mutual_info, dis)
    # calc_ptoq_edge_delete(labels, edges, mutual_info, dis)
    # exit(0)

    # calc_ptoq_edge_trainval(labels, edges, mutual_info, dis, train_mask, val_mask)
    exit(0)

    if train_mask is not None and val_mask is not None and test_mask is not None:
        idx_train = torch.LongTensor(torch.nonzero(torch.tensor(train_mask)).reshape(-1))
        idx_val = torch.LongTensor(torch.nonzero(torch.tensor(val_mask)).reshape(-1))
        idx_test = torch.LongTensor(torch.nonzero(torch.tensor(test_mask)).reshape(-1))

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    n = 5
    # edges = np.array([[1, 2], [0, 2], [2, 3], [3, 4], [0, 4], [1, 4]])
    edges = np.array([[1, 2], [0, 2], [2, 3], [3, 4]])
    m, dis = calc_dis_mutual_info(n, edges)
    print(m)
