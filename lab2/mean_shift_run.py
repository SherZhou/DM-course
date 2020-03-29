import lab2.mean_shift as ms
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 按行导入数据
def load_points(filename):
    data = genfromtxt(filename, delimiter=',', encoding='utf-8-sig')
    return data


def run():
    reference_points = load_points("iris.csv")
    iris = load_iris()
    target=iris.target
    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth=0.2)
    print('accuracy',accuracy_score(mean_shift_result.cluster_ids,target))
    print("Original Point     Shifted Point  Cluster ID")
    for i in range(len(mean_shift_result.shifted_points)):
        original_point = mean_shift_result.original_points[i]
        converged_point = mean_shift_result.shifted_points[i]
        cluster_assignment = mean_shift_result.cluster_ids[i]
        print("(%5.2f,%5.2f)  ->  (%5.2f,%5.2f)  cluster %i" % (
            original_point[0], original_point[1], converged_point[0], converged_point[1], cluster_assignment))


if __name__ == '__main__':
    run()

    data = np.genfromtxt('iris.csv', delimiter=',', encoding='utf-8-sig')
    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth=0.2)
    original_points = mean_shift_result.original_points
    shifted_points = mean_shift_result.shifted_points
    cluster_assignments = mean_shift_result.cluster_ids
    # 绘图
    x = original_points[:, 0]
    y = original_points[:, 1]
    Cluster = cluster_assignments
    centers = shifted_points
    fig = plt.figure()
    # 1×1网格，第一子图
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, c=Cluster, s=50)
    for i, j in centers:
        ax.scatter(i, j, s=50, c='red', marker='+')
    ax.set_xlabel('x:sepal length (cm)')
    ax.set_ylabel('y:sepal width (cm)')
    plt.colorbar(scatter)
    fig.savefig("mean_shift_result")
