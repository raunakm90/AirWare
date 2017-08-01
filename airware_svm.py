from data import Read_Data
import numpy as np
from sklearn.svm import SVC
import time
from utils.model_tuning import *
from utils.generate_report import *
import argparse

MODEL_PATH = "./baseline_models/svm/"


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
                                          brange=8, keras_format=False,
                                          plot_spectogram=False,
                                          baseline_format=True)

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
    grid_search_best_estimator = gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid,
                                                file_path=MODEL_PATH)

    print('It took ', time.time() - start, ' seconds.')

    return grid_search_best_estimator


def run_eval_svm():
    svm_clf_params = joblib.load(file_path + "clf_gridsearch.pkl")

    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', SVC())
    ])
    svm_clf_pipe = pipe.set_params(**svm_clf_params)
    start = time.time()
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False,
                                          plot_spectogram=False,
                                          baseline_format=True)
    eval_model(svm_clf_pipe, x, y, MODEL_PATH + "SVM")

    print('It took ', time.time() - start, ' seconds.')


def train_svm_loso():
    cv_obj = LeaveOneGroupOut()

    train_scores, test_scores = [], []
    y_true, y_hat = [], []
    class_names = []
    i = 1

    svm_clf_params = joblib.load(MODEL_PATH + "clf_gridsearch.pkl")
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', SVC())
    ])
    start = time.time()

    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False,
                                          plot_spectogram=False,
                                          baseline_format=True)

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
        y_hat_user = clf_pipe.predict(x_test_copy)  # Predictions per user
        class_names.append(lab_enc.classes_[y_test_copy])  # Class names per user
        y_hat.append(y_hat_user)  # Collect predictions for all users
        y_true.append(y_test_copy)  # Collect true values for all users

    y_true = flat_list_of_array(y_true)
    y_hat = flat_list_of_array(y_hat)

    # class_names = flat_list_of_array(class_names)
    print(classification_report(y_true=y_true, y_pred=y_hat, target_names=lab_enc.classes_, file_path=MODEL_PATH))
    print('It took ', time.time() - start, ' seconds.')
    return None


if __name__ == '__main__':
    function_map = {'gridSearch': run_gridSearch_svm,
                    'eval': run_eval_svm,
                    'train': train_svm_loso}
    parser = argparse.ArgumentParser(description="AirWare grid search and train model using different CV strategies")
    # "?" one argument consumed from the command line and produced as a single item
    # Positional arguments
    parser.add_argument('-function_name',
                        help="Define function to run for SVM",
                        choices=['gridSearch', 'eval',
                                 'train'])

    args = parser.parse_args()
    function = function_map[args.function_name]
    print("Running ", function)
    function()
