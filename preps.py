import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

def nan_check(df):
    missed = df.isna().sum().sum()
    if missed > 0:
        print("Количество пропущенных значений: ")
        for col in df.columns:
            if type(df[col].to_dict()[0]) == int or type(df[col].to_dict()[0]) == float:
                df[col].fillna(df[col].mean(), inplace = True)
            else:
                df[col].fillna(df[col].mode().iloc[0], inplace = True)
    else:
        print("Пропущенных значений нет")

def cat_features(df):
    flag = 0
    for col in df.columns:
        if not isinstance(df[col].to_dict()[0], (int, float)):
            flag = 1
            le = LabelEncoder()
            cat_col = le.fit_transform(df[col])
            df[col] = cat_col

    if flag == 0:
        print("Категориальных признаков нет")

    else:
        print("Категориальные признаки изменены")
    return df



def define_distrib(df):
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(df.columns):
        plt.subplot(3, 3, i + 1)
        df[col].plot(kind='hist')
        plt.title(df[col].name)
    plt.tight_layout()
    plt.show()

def leaf(y):
    y_n = list(y)
    val_leaf = max(y_n, key=y_n.count)
    return val_leaf
def min_max_scaler(df):
    #используется для равномерного распределения
    min_val = df.min()
    max_val = df.max()
    df = (df - min_val)/(max_val - min_val)


def splitter (X, y, test_size, random_state):
    rows = X.shape[0]
    ids = np.array(range(rows))
    random.seed(random_state)
    random.shuffle(ids)

    test = round(rows * test_size)

    test_ids = ids[0:test]
    tr_ids = ids[test:rows]

    X_train = pd.DataFrame(X.values[tr_ids, :], columns=X.columns)
    X_test = pd.DataFrame(X.values[test_ids, :], columns=X.columns)
    y_train = pd.DataFrame(y.values[tr_ids], columns=['success'])
    y_test = pd.DataFrame(y.values[test_ids], columns=['success'])
    return X_train, X_test, y_train, y_test

def create_metric(df):
    df['success'] = (df['GRADE'] >= 5).astype(int) #при 7 акураси 0.93

import random

def get_rand_ftr(col_names_list):
    ids = list(range(0, round(math.sqrt(len(col_names_list)))))
    random.shuffle(ids)
    col_chosen = []
    for i in ids:
        col_chosen.append(col_names_list[i])
    return col_chosen

def get_rand_frame(df):
    col_chosen = get_rand_ftr(df.columns)
    new_df = df[col_chosen]
    print("Количество признаков: ", len(col_chosen))
    print("Выбранные рандомно признаки", col_chosen)
    return new_df

