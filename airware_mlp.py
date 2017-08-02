from data import Read_Data
import numpy as np
from sklearn.neural_network import MLPClassifier
import time
from utils.model_tuning import *
from utils.generate_report import *
import argparse

MODEL_PATH = "./baseline_models/mlp/"


def run_gridSearch_mlp():
    start = time.time()
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False,
                                          plot_spectogram=False,
                                          baseline_format=True)

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
    grid_search_best_estimator = gridSearch_clf(x=x, y=y, groups=user, clf=clf_obj, param_grid=param_grid,
                                                file_path=MODEL_PATH)

    print('It took ', time.time() - start, ' seconds.')
    return grid_search_best_estimator


def run_eval_mlp():
    mlp_clf_params = joblib.load(MODEL_PATH + "clf_gridsearch.pkl")

    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', MLPClassifier())
    ])
    svm_clf_pipe = pipe.set_params(**mlp_clf_params)
    start = time.time()
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=8, keras_format=False,
                                          plot_spectogram=False,
                                          baseline_format=True)
    eval_model(svm_clf_pipe, x, y, MODEL_PATH + "MLP")

    print('It took ', time.time() - start, ' seconds.')


def train_mlp_loso():
    cv_obj = LeaveOneGroupOut()

    train_scores, test_scores = [], []
    y_true, y_hat = [], []
    class_names = []
    i = 1

    mlp_clf_params = joblib.load(MODEL_PATH + "clf_gridsearch.pkl")
    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', MLPClassifier())
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
        clf_pipe = pipe.set_params(**mlp_clf_params)

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
    function_map = {'gridSearch': run_gridSearch_mlp,
                    'eval': run_eval_mlp,
                    'train': train_mlp_loso}
    parser = argparse.ArgumentParser(description="AirWare grid search and train model using different CV strategies")
    # "?" one argument consumed from the command line and produced as a single item
    # Positional arguments
    parser.add_argument('-function_name',
                        help="Define function to run for MLP",
                        choices=['gridSearch', 'eval',
                                 'train'])

    args = parser.parse_args()
    function = function_map[args.function_name]
    print("Running ", function)
    function()
