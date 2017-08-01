import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics as mt
import pandas as pd

font = {'size': 12}

matplotlib.rc('font', **font)
sns.set(font_scale=1.5)


def flat_list_of_array(l):
    return np.concatenate(l).ravel()


# @TODO: Add mask to hide 0 entries in confusion matrix
def plot_confusion_matrix(cm, labels, title_text='Confusion Matrix', file_name="Confusion_Matrix.png"):
    plt.figure(figsize=(25, 8))
    acc = np.sum(cm.diagonal()) / np.sum(cm)
    cm_2 = cm / np.sum(cm, axis=1)[:, np.newaxis]
    cm_2 = np.append(cm_2, np.sum(cm, axis=1).reshape(len(labels), 1), axis=1)
    x_labels = np.append(labels, 'Support')
    sns.heatmap(cm_2, annot=True, fmt='.2f', vmin=0, vmax=1, xticklabels=x_labels, yticklabels=labels)
    plt.title(title_text, fontsize=20)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual Class', fontsize=15)
    plt.yticks(rotation=0)
    plt.savefig(file_name)


def plot_classification_report(report, file_name):
    plt.figure(figsize=(25, 8))
    sns.heatmap(report, annot=True, fmt='.2f', vmin=0, vmax=1)
    plt.title("Classification Report", fontsize=20)
    plt.xlabel('Metric', fontsize=15)
    plt.ylabel('Gesture Class', fontsize=15)
    plt.yticks(rotation=0)
    plt.savefig(file_name)


def classification_report(y_true, y_pred, target_names, file_path):
    precision, recall, f1, support = mt.precision_recall_fscore_support(y_true, y_pred)
    conf_mat = mt.confusion_matrix(y_true=y_true, y_pred=y_pred)
    per_class_acc = conf_mat.diagonal() / np.sum(conf_mat, axis=1)

    report_df = pd.DataFrame([precision, recall, f1, per_class_acc, support], columns=target_names).T
    report_df.columns = ['Precision', 'Recall', 'F1_Score', 'Accuracy', 'Support']

    plot_confusion_matrix(conf_mat, target_names, file_name=file_path + "Confusion_Matrix.png")
    plot_classification_report(report_df, file_name=file_path + "Classification_Report.png")

    return report_df


def write_results(train_sizes, train_scores, test_scores, file_path):
    print("Writing results...")
    np.savez(file_path + "_Train_Sizes.npz", train_sizes)
    np.savez(file_path + "_Train_Scores.npz", train_scores)
    np.savez(file_path + "_Test_Scores.npz", test_scores)
