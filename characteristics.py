def conf_matrix(y_test, y_pred):
    cm = {
        'TP': 0,  # True Positives
        'TN': 0,  # True Negatives
        'FP': 0,  # False Positives
        'FN': 0   # False Negatives
    }
    for true, pred in zip(y_test, y_pred):
        if true == 1 and pred == 1:
            cm['TP'] += 1
        elif true == 0 and pred == 0:
            cm['TN'] += 1
        elif true == 0 and pred == 1:
            cm['FP'] += 1
        elif true == 1 and pred == 0:
            cm['FN'] += 1
    return cm


def metrics(y_test, y_prend):
    cm = conf_matrix(y_test, y_prend)

    acc = (cm['TP'] + cm['TN']) / (cm['TP'] + cm['TN'] + cm['FN'] + cm['FP'])

    precision = cm['TP'] / (cm['TP'] + cm['FP'])

    recall = cm['TP'] / (cm['TP'] + cm['FN'])

    print(f'Accuracy: {acc} \nPrecision: {precision} \nRecall: {recall}')