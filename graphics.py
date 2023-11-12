import numpy as np
import matplotlib.pyplot as plt

def get_roc_auc(y_test, y_pred):
    sorted_indices = np.argsort(y_pred, kind='heapsort')[::-1]
    y_test = y_test[sorted_indices]

    num_positive = np.sum(y_test == 1)
    num_negative = np.sum(y_test == 0)

    tpr = [0]
    fpr = [0]

    tp = 0
    fp = 0

    for i in range(len(y_test)):
        if y_test[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr.append(tp / num_positive)
        fpr.append(fp / num_negative)

    roc_auc = np.trapz(tpr, fpr)
    return fpr, tpr, roc_auc

def get_pr_auc(y_test, y_pred):
    sorted_indices = np.argsort(y_pred, kind='heapsort')[::-1]
    y_test = y_test[sorted_indices]

    precision = np.zeros(len(y_test) + 1)
    recall = np.zeros(len(y_test) + 1)

    tp = 0
    fp = 0

    for i in range(len(y_test)):
        if y_test[i] == 1:
            tp += 1
        else:
            fp += 1

        precision[i + 1] = tp / (tp + fp)
        recall[i + 1] = tp / np.sum(y_test == 1)

    pr_auc = np.trapz(precision, recall)
    return precision, recall, pr_auc
def draw_plots(true_labels, predicted_scores):
    fpr, tpr, roc_auc = get_roc_auc(true_labels, predicted_scores)
    precision, recall, pr_auc = get_pr_auc(true_labels, predicted_scores)

    #первая кривая
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    #вторая кривая
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()
