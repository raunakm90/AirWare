from data import Read_Data
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
import sklearn.metrics as mt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import time


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

    print("Saving best estimator to disk")
    joblib.dump(grid.best_estimator_, filename=file_path + "clf_gridsearch.pkl")

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
                                          brange=8, keras_format=False)
    clf_obj = SVC()
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
    gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid, file_path="./baseline_models/svm/")

    print('It took ', time.time() - start, ' seconds.')


def run_gridSearch_mlp():
    start = time.time()
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False)

    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    # Create a mask for PCA only on doppler signature
    mask = np.arange(x.shape[1]) < x.shape[1] - 2

    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', MLPClassifier())
    ])
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


if __name__ == '__main__':
    # clf = joblib.load("./baseline_models/svm/svm_gridSearch.pkl")
    # print(clf)
    # run_gridSearch_mlp()

    clf = joblib.load("./baseline_models/mlp/mlp_gridSearch.pkl")
    print(clf)
