import numpy as np
import math as m

class DecisionTree:
    def __init__(self, level=0):
        self.level = 0
        self.featureName = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # 检查纯度是否大于0.95
    def check_purity(self, data):

        labels_values = [int(i) for i in data[:, -1]]
        unique_class, counts = np.unique(labels_values, return_counts=True)
        purity = counts / counts.sum()
        print('purity',purity)
        a=purity>=0.95
        if (a.any()==True):
            return True
        else:
            return False

    # 得到可能的所有分割点，返回potential_splits字典
    def get_potential_split(self, data):
        potential_splits = {}
        n_columns = len(data[0]) - 1
        for c in range(n_columns):
            potential_splits[c] = []
            values = data[:, c]
            unique_values = np.unique(values)

            for i in range(1, len(unique_values)):
                ps = (unique_values[i] + unique_values[i - 1]) / 2
                potential_splits[c].append(ps)
        return potential_splits

    # 分割数据（是否满足分割条件），返回左子树，右子树数据
    def split_data(self, data, split_column, split_value):
        split_column_values = data[:, split_column]

        data_left = data[split_column_values <= split_value]
        data_right = data[split_column_values > split_value]

        return data_left, data_right
    # 信息熵
    def entropy(self, data):
        labels_values = [int(i) for i in data[:, -1]]
        unique_class, counts = np.unique(labels_values, return_counts=True)
        probabilities = counts / counts.sum()
        entropies = sum(probabilities * -np.log2(probabilities))
        return entropies
    def cross_entropy(self, data_left, data_right):
        n_total = len(data_left) + len(data_right)
        p_left = len(data_left) / n_total
        p_right = len(data_right) / n_total
        return (p_left * self.entropy(data_left) + p_right * self.entropy(data_right))
    # 信息增益
    def gain(self, data, data_left, data_right):
        ig = self.entropy(data) - self.cross_entropy(data_left, data_right)
        return (ig)

    # 找最佳分割点 返回最佳分割属性与最佳分割点
    def find_best_split(self, data, potential_splits):
        max_entropy = m.inf
        max_gain = -m.inf

        for c in potential_splits:
            for v in potential_splits[c]:
                data_left, data_right = self.split_data(data, split_column=c, split_value=v)
                current_entropy = self.cross_entropy(data_left, data_right)
                current_gain = self.gain(data, data_left, data_right)

                if current_entropy < max_entropy:
                    max_gain = current_gain
                    max_entropy = current_entropy
                    best_split_column = c
                    best_split_value = v
        return best_split_column, best_split_value

    # data = Xtrain + Ytrain(instance+target)
    # f = features
    # level = current level

    def decision_tree(self, data, f, level):
        # 纯度>95%
        if self.check_purity(data):
            labels_values = [int(i) for i in data[:, -1]]
            
            unique_class, counts = np.unique(labels_values, return_counts=True)
            print('Level', level)
            for i in range(len(counts)):
                cla = unique_class, i
                print('Count of', cla, '= ', counts[i])
                # print('Count of', i, '= ', counts[i])
                print('Current Entropy  is = ', self.entropy(data))
                print('Reached leaf Node')
            classification = unique_class[counts.argmax()]
            print('classification:',classification)

        # 数量<=5
        elif len(data)<=5:
            labels_values = [int(i) for i in data[:, -1]]
            unique_class, counts = np.unique(labels_values, return_counts=True)
            print('Level', level)
            for i in range(len(counts)):
                cla = unique_class, i
                print('Count of', cla, '= ', counts[i])
                # print('Count of', i, '= ', counts[i])
                print('Current Entropy  is = ', self.entropy(data))
                print('Reached leaf Node')
            classification = unique_class[counts.argmax()]
            print('classification:', classification)

        # features==0
        elif len(f) == 0:
            labels_values = [int(i) for i in data[:, -1]]
            unique_class, counts = np.unique(labels_values, return_counts=True)
            print('Level', level)
            for i in range(len(counts)):
                cla = unique_class, i
                print('Count of', cla, '= ', counts[i])
                # print('Count of', i, '= ', counts[i])
                print('Current Entropy  is = ', self.entropy(data))
                print('Reached leaf Node')
            classification = unique_class[counts.argmax()]
            print('classification:', classification)

        # 不满足条件，迭代分割，最后返回所有子树构成决策树
        else:
            potential_split = self.get_potential_split(data)

            best_split_feature, best_split_value = self.find_best_split(data, potential_split)
            data_left, data_right = self.split_data(data, best_split_feature, best_split_value)

            labels_values = [int(i) for i in data[:, -1]]
            unique_class, counts = np.unique(labels_values, return_counts=True)
            print('Level', level)
            for i in range(len(counts)):
                cla = unique_class, i
                print('Count of', cla, '= ', counts[i])
                # print('Count of', i, '= ', counts[i])
                # print('Purity of', cla, '= ')
                print('Current Entropy  is = ', self.entropy(data))
                print('Splitting on feature', f[best_split_feature],"<=",best_split_value, 'with gain', self.gain(data, data_left, data_right))

            data_left = np.delete(data_left, best_split_feature, 1)
            data_right = np.delete(data_right, best_split_feature, 1)

            que = '{} <= {}'.format(f[best_split_feature], best_split_value)
            sub_tree = {que: []}

            f.remove(f[best_split_feature])
            y = self.decision_tree(data_left, f, level + 1)
            n = self.decision_tree(data_right, f, level + 1)

            sub_tree[que].append(y)
            sub_tree[que].append(n)
            return sub_tree

    #  将数据调整为能放入决策树函数的形式
    def fit(self, X_train, Y_train):
        Y_train = Y_train.reshape(len(Y_train), 1)
        self.data = np.concatenate((X_train, Y_train), axis=1)
        self.dictionary = self.decision_tree(self.data, self.featureName, self.level)