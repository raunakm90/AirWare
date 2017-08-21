import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics as mt
import pandas as pd

font = {'size': 12}

matplotlib.rc('font', **font)
sns.set(font_scale=1.5)


def write_results(train_sizes, train_scores, test_scores, file_path):
    print("Writing results...")
    np.savez(file_path + "_Train_Sizes.npz", train_sizes)
    np.savez(file_path + "_Train_Scores.npz", train_scores)
    np.savez(file_path + "_Test_Scores.npz", test_scores)


# Training history for DL models
def plot_training_curve(file_path):
    train_loss = np.load(file_path + "_Train_Loss.npz")['arr_0']
    val_loss = np.load(file_path + "_Val_Loss.npz")['arr_0']

    train_acc = np.load(file_path + "_Train_Acc.npz")['arr_0']
    val_acc = np.load(file_path + "_Val_Acc.npz")['arr_0']

    # Average and SD for loss across users
    train_loss_avg = np.mean(train_loss, axis=0)
    val_loss_avg = np.mean(val_loss, axis=0)
    train_loss_std = np.std(train_loss, axis=0)
    val_loss_std = np.std(val_loss, axis=0)

    # Average and Loss for accuracy across users
    train_acc_avg = np.mean(train_acc, axis=0)
    val_acc_avg = np.mean(val_acc, axis=0)
    train_acc_std = np.std(train_acc, axis=0)
    val_acc_std = np.std(val_acc, axis=0)

    nb_epoch = np.arange(0, len(train_loss_avg))

    fig, axarr = plt.subplots(1, 2, sharey=False, figsize=(25, 8))
    plt.suptitle("Training Performance", fontsize=20)

    axarr[0].set_xlabel("Number of Iterations")
    axarr[0].set_ylabel("Cross-Entropy Loss")
    axarr[0].set_title("Model Performance: Cross-Entropy Loss")
    axarr[0].fill_between(nb_epoch, train_loss_avg - train_loss_std,
                          train_loss_avg + train_loss_std, alpha=0.1,
                          color="r")
    axarr[0].fill_between(nb_epoch, val_loss_avg - val_loss_std,
                          val_loss_avg + val_loss_std, alpha=0.1, color="g")
    axarr[0].plot(nb_epoch, train_loss_avg, color="r")
    axarr[0].plot(nb_epoch, val_loss_avg, color="g")

    axarr[1].set_xlabel("Number of Iterations")
    axarr[1].set_ylabel("Accuracy")
    axarr[1].set_title("Model Performance: Accuracy")
    axarr[1].fill_between(nb_epoch, train_acc_avg - train_acc_std,
                          train_acc_avg + train_acc_std, alpha=0.1,
                          color="r")
    axarr[1].fill_between(nb_epoch, val_acc_avg - val_loss_std,
                          val_acc_avg + val_acc_std, alpha=0.1, color="g")
    axarr[1].plot(nb_epoch, train_acc_avg, color='r')
    axarr[1].plot(nb_epoch, val_acc_avg, color='g')

    plt.savefig(file_path + "Training_Performance.png")

    return train_acc, val_acc


def plot_training_curve_baseline(train_scores, test_scores, train_sizes, subplot_titles):
    fig, axarr = plt.subplots(1, 2, sharey=True, figsize=(25, 8))
    plt.suptitle("Baseline Models Training Performance", fontsize=20)

    for i in range(len(train_scores)):
        axarr[i].grid()
        axarr[i].set_xlabel("Training examples")
        axarr[i].set_ylabel("Score")
        train_scores_mean = np.mean(train_scores[i], axis=1)
        train_scores_std = np.std(train_scores[i], axis=1)
        test_scores_mean = np.mean(test_scores[i], axis=1)
        test_scores_std = np.std(test_scores[i], axis=1)

        axarr[i].fill_between(train_sizes[i], train_scores_mean - train_scores_std,
                              train_scores_mean + train_scores_std, alpha=0.1,
                              color="r")
        axarr[i].fill_between(train_sizes[i], test_scores_mean - test_scores_std,
                              test_scores_mean + test_scores_std, alpha=0.1, color="g")
        axarr[i].plot(train_sizes[i], train_scores_mean, 'o-', color="r",
                      label="Training Score")
        axarr[i].plot(train_sizes[i], test_scores_mean, 'o-', color="g",
                      label="Cross-Validation Score")
        axarr[i].legend(loc="best")
        axarr[i].set_title(subplot_titles[i])
    return plt



def write_results_models(train_scores, test_scores, class_names, y_pred, y_true, file_path):
    print("Writing results...")
    np.savez(file_path + "_Train_Scores.npz", train_scores)
    np.savez(file_path + "_Test_Scores.npz", test_scores)
    np.savez(file_path + "_Predictions.npz", y_pred)
    np.savez(file_path + "_Truth.npz", y_true)
    np.savez(file_path + "_Class_Names.npz", class_names)


def write_train_hist(train_val_hist, file_path):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for item in train_val_hist:
        train_loss.append(item.history['loss'])
        val_loss.append(item.history['val_loss'])
        train_acc.append(item.history['acc'])
        val_acc.append(item.history['val_acc'])
    np.savez(file_path + "_Train_Loss.npz", train_loss)
    np.savez(file_path + "_Train_Acc.npz", train_acc)
    np.savez(file_path + "_Val_Loss.npz", val_loss)
    np.savez(file_path + "_Val_Acc.npz", val_acc)
