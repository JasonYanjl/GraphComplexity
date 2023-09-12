from pygcn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    n = 5000
    output_dict = {}
    for k in range(1):
        tmp_one = np.triu(np.ones((n, n)), 1)
        tmp_list = np.where(tmp_one > 0)
        Edges = np.concatenate((np.expand_dims(tmp_list[0],axis=0), np.expand_dims(tmp_list[1],axis=0))).T
        output_dict['edges'] = list(range(0, Edges.shape[0], 50000))
        np.random.shuffle(Edges)
        res = []
        calc_complexity_fast(n, Edges[:, :])
        for i in range(0, Edges.shape[0], 50000):
            print(i)
            res.append(calc_complexity_fast(n, Edges[:i, :]))

        output_dict[f'res_{k}'] = res

    data_frame = pd.DataFrame(output_dict)
    data_frame.to_csv(f'Result{n}_5.csv', index=None)


def paint_trend():
    n = 50
    csv_path = os.path.join('.', 'complexity', f'Result{n}.csv')
    csv_file = pd.read_csv(csv_path)

    edge = csv_file['edges_normal'].tolist()
    avg_acc = np.array(csv_file['avg_normal'].tolist())

    max_index = np.argmax(avg_acc)

    max_acc = np.array(csv_file['max_normal'].tolist())
    min_acc = np.array(csv_file['min_normal'].tolist())

    max_acc = (max_acc - avg_acc) * 5 + avg_acc
    min_acc = (min_acc - avg_acc) * 5 + avg_acc

    plt.ylim((0, 1.1))
    plt.plot(edge, avg_acc, linewidth = 1)
    plt.fill_between(edge, max_acc, min_acc, alpha = 0.25)

    plt.scatter(edge[max_index], avg_acc[max_index], 20, color='red')
    plt.plot([edge[max_index], edge[max_index], ], [0, avg_acc[max_index], ], 'k--', linewidth=1)
    plt.annotate(f'max complexity: x={edge[max_index]}', xy=(edge[max_index], avg_acc[max_index]), xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

    plt.title(f'n = {n}')
    plt.xlabel('#edges / (n*(n-1)/2)')
    plt.ylabel('complexity / #max_complexity')

    plt.savefig(f'n_{n}.png')
    plt.show()


if __name__ == "__main__":
    # main()
    paint_trend()