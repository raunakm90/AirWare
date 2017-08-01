from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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