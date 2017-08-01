from data import Read_Data
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, validation_curve, learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.metrics as mt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

font = {'size': 12}

matplotlib.rc('font', **font)
sns.set(font_scale=1.5)


# Reference - https://stackoverflow.com/questions/24124622/pipeline-with-pca-on-feature-subset-only-in-scikit-learn
class MaskedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, mask=None):
        """
        Computes PCA on subset of data and combines PC's with the remaining data.
        :param n_components: Number of principal components
        :param mask: List of boolean values to select columns
        """
        self.n_components = n_components
        self.mask = mask
        self.pca = None

    def fit(self, X, y):
        self.pca = PCA(n_components=self.n_components)
        mask = self.mask if self.mask is not None else slice(None)
        self.pca.fit(X[:, mask])
        return self

    def transform(self, X):
        mask = self.mask if self.mask is not None else slice(None)
        pca_transformed = self.pca.transform(X[:, mask])
        if self.mask is not None:
            rem_cols = X[:, ~mask]
            return np.hstack([pca_transformed, rem_cols])
        else:
            return pca_transformed

    def inverse_transform(self, X):
        """

        :param X: array-like, shape(n_samples, n_components)
        :return: X_original (n_samples, n_features)
        """
        if self.mask is not None:
            # Inverse transformation of pca data
            inv_mask = np.arange(len(X[0])) >= sum(~self.mask)
            inv_transformed = self.pca.inverse_transform(X[:, inv_mask])

            # Place inverse transformed columns back in their original order
            inv_transformed_reorder = np.zeros([len(X), len(self.mask)])
            inv_transformed_reorder[:, self.mask] = inv_transformed
            inv_transformed_reorder[:, ~self.mask] = X[:, ~inv_mask]
            return inv_transformed_reorder
        else:
            return self.pca.inverse_transform(X)

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
    plt.xlabel('Gesture Class', fontsize=15)
    plt.ylabel('Metric', fontsize=15)
    plt.yticks(rotation=0)
    plt.savefig(file_name)


def classification_report(y_true, y_pred, target_names, file_path):
    precision, recall, f1, support = mt.precision_recall_fscore_support(y_true, y_pred)
    conf_mat = mt.confusion_matrix(y_true=y_true, y_pred=y_pred)
    per_class_acc = conf_mat.diagonal() / np.sum(conf_mat, axis=1)

    report_df = pd.DataFrame([precision, recall, f1, per_class_acc, support], columns=target_names).T
    report_df.columns = ['Precision', 'Recall', 'F1_Score', 'Accuracy', 'Support']

    plot_confusion_matrix(conf_mat, target_names, file_name=file_path+"Confusion_Matrix.png")
    plot_classification_report(report_df, file_name=file_path+"Classification_Report.png")

    return report_df


# Returns indices of columns that have sd less than or equal to thresh
def remove_near_zero_var(x, thresh=20):
    std_devs = np.std(x, axis=0)
    return np.where(std_devs <= thresh)[0]


def gridSearch_clf(x, y, groups, param_grid, clf, file_path='./baseline_models/'):
    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    # Define Leave one subject out CV object
    cv_obj = LeaveOneGroupOut()

    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', clf)
    ])

    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv_obj, n_jobs=-1,
                        scoring='accuracy', verbose=1)
    grid.fit(x, y, groups=groups)

    print("Best Score")
    print(grid.best_score_)

    print("Grid Search based Best Estimator")
    print(grid.best_estimator_)

    print("Saving best estimator to disk")
    joblib.dump(grid.best_params_, filename=file_path + "clf_gridsearch.pkl")

    return grid.best_estimator_


def run_gridSearch_svm():
    start = time.time()
    # Number of principle components for Masked PCA
    n_components_range = [100, 200]
    # C trades off misclassification of training examples against simplicity of the decision surface.
    # Higher C selects more samples as support vectors
    c_range = np.logspace(-3, 3, 7)
    # gamma defines how far the influence of a single training example reaches; low==far.
    # Inverse of the radius of influence of samples selected by the model as support vectors
    gamma_range = np.logspace(-3, 3, 7)
    kernel_options = ['rbf', 'linear']

    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False, plot_spectogram=False, baseline_format=True)
    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    # Create a mask for PCA only on doppler signature
    mask = np.arange(x.shape[1]) < x.shape[1] - 2
    param_grid = [
        {
            'reduce_dim__n_components': n_components_range,
            'reduce_dim__mask': [mask],
            'classify__C': c_range,
            'classify__kernel': kernel_options,
            'classify__gamma': gamma_range,
            'classify__class_weight': ['balanced']
        }
    ]
    clf_obj = SVC()
    gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid, file_path="./baseline_models/svm/")

    print('It took ', time.time() - start, ' seconds.')


def run_gridSearch_mlp():
    start = time.time()
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False, plot_spectogram=False, baseline_format=True)

    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    # Create a mask for PCA only on doppler signature
    mask = np.arange(x.shape[1]) < x.shape[1] - 2

    # Number of principle components for Masked PCA
    n_components_range = [100, 200]

    param_grid = {
        'reduce_dim__n_components': n_components_range,
        'reduce_dim__mask': [mask],
        'classify__hidden_layer_sizes': [(50,), (100,), (500,), (50, 50,), (100, 50,), (100, 20,), (50, 20,),
                                         (500, 100,),
                                         (500, 250,), (250, 100,)],
        'classify__alpha': [0.0001, 0.01, 1, 10, 100],
        'classify__activation': ['relu', 'tanh', 'logistic'],
        'classify__max_iter': [1000],
        'classify__random_state': [576],
        'classify__solver': ['adam'],
        'classify__early_stopping': [True],
        'classify__beta_1': [0.9],
        'classify__beta_2': [0.999]}

    clf_obj = MLPClassifier()
    gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid, file_path="./baseline_models/mlp/")

    print('It took ', time.time() - start, ' seconds.')


# Reference - http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
def plot_validation_curve(train_scores, test_scores, param_range, file_path):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("Misclassification Cost - C")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig(file_path)


# Reference - http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(train_sizes, train_scores, test_scores, title, file_path, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(file_path)


def write_results(train_sizes, train_scores, test_scores, file_path):
    print("Writing results...")
    np.savez(file_path + "_Train_Sizes.npz", train_sizes)
    np.savez(file_path + "_Train_Scores.npz", train_scores)
    np.savez(file_path + "_Test_Scores.npz", test_scores)


def eval_model(clf_pipe, x, y, file_path):
    cv_obj = StratifiedShuffleSplit(n_splits=5, random_state=234, test_size=0.3)
    train_sizes = np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(clf_pipe, x, y, cv=cv_obj,
                                                            n_jobs=-1, train_sizes=train_sizes)
    write_results(train_sizes, train_scores, test_scores, file_path)
    # plot_learning_curve(train_sizes, train_scores, test_scores, "SVM_Learning_Curve",
    #                     file_path)


def run_eval_svm():
    svm_clf_params = joblib.load("./baseline_models/svm/clf_gridsearch.pkl")
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', SVC())
    ])
    svm_clf_pipe = pipe.set_params(**svm_clf_params)
    start = time.time()
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False, plot_spectogram=False, baseline_format=True)

    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)
    cv_obj = LeaveOneGroupOut()
    # param_range = np.logspace(-5, 2, 8)
    # train_scores, test_scores = validation_curve(svm_clf_pipe, x, y, groups=user,
    #                                              param_name='classify__C', param_range=param_range,
    #                                              cv=cv_obj, scoring='accuracy', n_jobs=-1)
    # plot_validation_curve(train_scores, test_scores, param_range, "./baseline_models/svm/Validation_Curve.png")

    eval_model(svm_clf_pipe, x, y, "./baseline_models/svm/SVM")

    print('It took ', time.time() - start, ' seconds.')



def train_svm_loso():
    file_path = "./baseline_models/svm/"
    cv_obj = LeaveOneGroupOut()

    train_scores, test_scores = [], []
    y_true, y_hat = [], []
    class_names = []
    i = 1

    svm_clf_params = joblib.load(file_path+"clf_gridsearch.pkl")
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', SVC())
    ])
    start = time.time()

    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False)

    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    for train_idx, test_idx in cv_obj.split(x, y, user):
        print("\nUser:", i)
        i += 1
        # Train and test data - leave one subject out
        x_train, y_train = x[train_idx, :], y[train_idx]
        x_test, y_test = x[test_idx, :], y[test_idx]

        # Create copies of the train and test data sets
        x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
        x_test_copy, y_test_copy = x_test.copy(), y_test.copy()

        # Call model function - Refit a new model
        clf_pipe = pipe.set_params(**svm_clf_params)

        # Fit model
        clf_pipe.fit(x_train_copy, y_train_copy)
        # Evaluate training scores
        train_scores.append(clf_pipe.score(x_train_copy, y_train_copy))
        # Evaluate test scores
        test_scores.append(clf_pipe.score(x_test_copy, y_test_copy))

        # Predict for test data
        y_hat_user = clf_pipe.predict(x_test_copy)
        class_names.append(lab_enc.classes_[y_test_copy])
        y_hat.append(y_hat_user)
        y_true.append(y_test_copy)

    print(len(y_true))
    y_true = flat_list_of_array(y_true)
    y_hat = flat_list_of_array(y_hat)

    # class_names = flat_list_of_array(class_names)
    print(classification_report(y_true=y_true, y_pred=y_hat, target_names=lab_enc.classes_, file_path=file_path))
    print('It took ', time.time() - start, ' seconds.')
    return None


if __name__ == '__main__':
    # run_gridSearch_mlp()
    # run_gridSearch_svm()
    run_eval_svm()
    # train_svm_loso()
