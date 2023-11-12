import numpy as np
from preps import *


class Tree_Node(): #value для листа, остальное для десижн нод
    def __init__(self, value = None, children = None, feature_idx = None, feature_limits = None, info = None):
        self.value = value
        self.children = children
        self.feature_idx = feature_idx #indeks dlya priznaka
        self.feature_limits = feature_limits
        self.info = info


class Not_Binary_Decision_Tree():
    def __init__(self, min_samples_s = 2):
        self.min_samples_s = min_samples_s
        self.root = None
    def fit(self, X, y):
        df = np.concatenate([X, y], axis=1)
        self.root = self.create_tree(df)

    def create_tree(self, df, current_level = 0):
        rows = df.shape[0]
        n_features = df.shape[1] - 1
        if rows >= self.min_samples_s:
            splitted_tree = self.get_best_split(df, n_features)
            if splitted_tree['info'] > 0.0:
                childrensy = splitted_tree['children']
                childrensy_dereva = {}
                for key in list(childrensy.keys()):
                    child = childrensy[key]
                    child_node = self.create_tree(child, current_level + 1)
                    childrensy_dereva[key] = child_node
                return Tree_Node(None, childrensy_dereva, splitted_tree['feature_idx'], splitted_tree['feature_limits'], splitted_tree['info'])
        leaf_val = leaf(df[:, -1])
        return Tree_Node(value=leaf_val)



    def get_best_split(self, df, n_features):
        splitted = {}
        info_max = -1000000
        for idx in range(n_features):
            vals = df[:, idx]
            lims = np.unique(vals)
            children = {}
            children_y = {}
            for lim in lims:
                child = np.array([ftr for ftr in df if float(ftr[idx]) == float(lim)])
                child_y = child[:, -1]
                children[f'{lim}'] = child
                children_y[f'{lim}'] = child_y
            y = df[:, -1]
            info_val = self.count_info_gain(y, children_y)
            if info_val > info_max:
                splitted['children'] = children
                splitted['feature_idx'] = idx
                splitted['feature_limits'] = lims
                splitted['info'] = info_val
                info_max = info_val
        return splitted



    def count_info_gain(self, y, children_y):
        ent_y = self.get_entropy(y)
        childrens_ents = 0
        for key in list(children_y.keys()):
            ent_child = self.get_entropy(children_y[key])
            childrens_ents -= ent_child
        result = ent_y + childrens_ents
        return result

    def get_entropy(self, y):
        uniqs = np.unique(y)
        res = 0
        for el in uniqs:
            is_in = len(y[y == el]) / len(y)
            res += is_in * np.log2(is_in)
        return res

    def predictor(self, x, tree):
        if tree.value is None:
            val = x[tree.feature_idx]
            return self.predictor(x, tree.children[str(val)])
        return tree.value

    def predict(self, X):
        y_pred = [self.predictor(x, self.root) for x in X]
        return y_pred
