import pandas as pd
import numpy as np
from preps import *

class Tree_Node(): #value для листа, остальное для десижн нод
    def __init__(self, value = None, left = None, right = None, feature_idx = None,
                 feature_limit = None, info = None):
        self.value = value
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.feature_limit = feature_limit
        self.info = info


class Binary_Decision_Tree():
    def __init__(self, max_depth = 2, min_samples_s = 1):
        self.max_depth = max_depth
        self.min_samples_s = min_samples_s
        self.root = None

    #разделение деревьев на поддеревья по принципу наибольшей инфы
    def tree_splitter(self, df, feature_cnt):
        splitted = {}
        info_max = -1000000
        for idx in range(feature_cnt):
            vals = df[:, idx]
            lims = np.unique(vals)
            for lim in lims:
                left = np.array([ftr for ftr in df if float(ftr[idx]) <= float(lim)])
                right = np.array([ftr for ftr in df if float(ftr[idx]) > float(lim)])
                if len(left) != 0 and len(right) != 0:
                    y = df[:, -1]
                    left_y = left[:, -1]
                    right_y = right[:, -1]
                    info_val = self.info_count(y, left_y, right_y)
                    if info_val > info_max:
                        splitted['left'] = left
                        splitted['right'] = right
                        splitted['feature_idx'] = idx
                        splitted['feature_limit'] = lim
                        splitted['info'] = info_val
                        info_max = info_val

        return splitted

    #расчет количества информации через энтропию
    def info_count(self, parent, c_l, c_r):
        uniq_p = np.unique(parent)
        uniq_c_l = np.unique(c_l)
        uniq_c_r = np.unique(c_r)
        ent_p = 0
        ent_c_l = 0
        ent_c_r = 0
        for el in uniq_p:
            pres_class = len(parent[parent == el]) / len(parent)
            ent_p += -pres_class * np.log2(pres_class)
        for el in uniq_c_l:
            pres_class = len(c_l[c_l == el]) / len(c_l)
            ent_c_l += -pres_class * np.log2(pres_class)
        for el in uniq_c_r:
            pres_class = len(c_r[c_r == el]) / len(c_r)
            ent_c_r += -pres_class * np.log2(pres_class)
        return ent_p - (len(c_l) / len(parent)) * ent_c_l - (len(c_r) / len(parent)) * ent_c_r

    #само создание дерева
    def create_tree(self, df, current_level = 0):
        rows = df.shape[0]
        features_cnt = df.shape[1] - 1

        if rows >= self.min_samples_s and current_level <= self.max_depth:
            splitted = self.tree_splitter(df, features_cnt)

            if splitted['info'] > 0.0:
                tree_left = self.create_tree(splitted['left'], current_level + 1)
                tree_right = self.create_tree(splitted['right'], current_level + 1)

                return Tree_Node(None, tree_left, tree_right, splitted['feature_idx'], splitted['feature_limit'],
                                 splitted['info'])
        leaf_val = leaf(df[:, -1])
        return Tree_Node(value=leaf_val)

    #фит (склейка хар-тик и таргета в датасет, объявление корня
    def fit(self, X, y):
        df = np.concatenate([X, y], axis = 1)
        self.root = self.create_tree(df)

    #предсказательная функция
    def predictor(self, x, trained_tree):
        if trained_tree.value == None:
            val = x[trained_tree.feature_idx]
            if val <= trained_tree.feature_limit:
                return self.predictor(x, trained_tree.left)
            else:
                return self.predictor(x, trained_tree.right)

        return trained_tree.value
    #само предсказание
    def predict(self, X):
        y_pred = [self.predictor(x, self.root) for x in X]
        return y_pred