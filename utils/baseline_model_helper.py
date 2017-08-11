import sys

sys.path.append('D:/AirWare/')
from data import Read_Data
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedShuffleSplit, learning_curve, \
    train_test_split
from .feature_engineering import remove_near_zero_var, MaskedPCA
from .generate_report import write_results, write_results_models, flat_list_of_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time

CV_FOLDS = 5
MODEL_PATH = './baseline_models/'


def airware_baseline_data():
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, lab_enc = gd.compile_data(nfft=4096, overlap=0.5,
                                          brange=16, keras_format=False,
                                          plot_spectogram=False,
                                          baseline_format=True)
    return x, y, user, lab_enc


def gridSearch_clf(x, y, groups, param_grid, clf, file_path=MODEL_PATH):
    # Define Leave one subject out CV object
    cv_obj = LeaveOneGroupOut()

    pipe = Pipeline([
        ('normalize', StandardScaler()),
        ('reduce_dim', MaskedPCA()),
        ('classify', clf)
    ])

    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv_obj, n_jobs=6,
                        scoring='accuracy', verbose=1)
    grid.fit(x, y, groups=groups)

    print("Best Score")
    print(grid.best_score_)

    print("Grid Search based Best Estimator")
    print(grid.best_estimator_)

    print("Saving best estimator to disk")
    joblib.dump(grid.best_params_, filename=file_path + "clf_gridsearch.pkl")

    return grid.best_estimator_


def eval_model(clf_pipe, x, y, file_path):
    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    cv_obj = StratifiedShuffleSplit(n_splits=5, random_state=234, test_size=0.3)
    train_sizes = np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(clf_pipe, x, y, cv=cv_obj,
                                                            n_jobs=-1, train_sizes=train_sizes)
    write_results(train_sizes, train_scores, test_scores, file_path)
    # plot_learning_curve(train_sizes, train_scores, test_scores, "SVM_Learning_Curve",
    #                     file_path)


def train_clf_loso(pipe, clf_params, model_path=MODEL_PATH):
    cv_obj = LeaveOneGroupOut()

    train_scores, test_scores = [], []
    y_true, y_pred = [], []
    i = 1
    start = time.time()

    x, y, user, lab_enc = airware_baseline_data()

    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    for train_idx, test_idx in cv_obj.split(x, y, user):
        print("User:", i)
        i += 1
        # Train and test data - leave one subject out
        x_train, y_train = x[train_idx, :], y[train_idx]
        x_test, y_test = x[test_idx, :], y[test_idx]

        # Create copies of the train and test data sets
        x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
        x_test_copy, y_test_copy = x_test.copy(), y_test.copy()

        # Call model function - Refit a new model
        clf_pipe = pipe.set_params(**clf_params)

        # Fit model
        clf_pipe.fit(x_train_copy, y_train_copy)
        # Evaluate training scores/accuracy
        train_scores.append(clf_pipe.score(x_train_copy, y_train_copy))
        # Evaluate test scores/accuracy
        test_scores.append(clf_pipe.score(x_test_copy, y_test_copy))

        # Predict for test data
        y_pred_user = clf_pipe.predict(x_test_copy)
        y_pred.append(y_pred_user)  # Collect predictions for all users
        y_true.append(y_test_copy)  # Collect true values for all users

    # y_true = flat_list_of_array(y_true)
    # y_hat = flat_list_of_array(y_hat)
    write_results_models(train_scores=train_scores, test_scores=test_scores, class_names=lab_enc.classes_,
                         y_pred=y_pred, y_true=y_true, file_path=model_path)
    print('It took ', time.time() - start, ' seconds.')
    return None


def train_clf_personalized(pipe, clf_params, model_path=MODEL_PATH):
    cv_obj = LeaveOneGroupOut()
    start = time.time()
    x, y, user, lab_enc = airware_baseline_data()

    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    i = 0  # Keep track of users

    train_scores, test_scores = [], []
    y_true, y_pred = [], []

    for rem_idx, keep_idx in cv_obj.split(x, y, user):
        print("\nUser:", i)
        i += 1

        # Keep only the user data
        # x_rem, y_rem = x[rem_idx,:,:,:], y[rem_idx]
        x_keep, y_keep = x[keep_idx, :], y[keep_idx]
        train_val_scores, test_val_scores = [], []
        y_true_val, y_pred_val = [], []

        # Define CV object
        cv_strat = StratifiedShuffleSplit(n_splits=CV_FOLDS, test_size=0.4, random_state=i * 243)
        j = 0  # Keep track of CV Fold

        # Stratified cross validation for each user
        for train_idx, test_idx in cv_strat.split(x_keep, y_keep):
            print('Fold:', str(j))
            x_train, y_train = x_keep[train_idx, :], y_keep[train_idx]
            x_test, y_test = x_keep[test_idx, :], y_keep[test_idx]

            x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
            x_test_copy, y_test_copy = x_test.copy(), y_test.copy()

            # Call model function - Refit a new model
            clf_pipe = pipe.set_params(**clf_params)
            # Fit model
            clf_pipe.fit(x_train_copy, y_train_copy)
            # Predictions
            y_pred_user = clf_pipe.predict(x_test_copy)
            # Evaluate training scores
            train_val_scores.append(clf_pipe.score(x_train_copy, y_train_copy))
            # Evaluate test scores
            test_val_scores.append(clf_pipe.score(x_test_copy, y_test_copy))
            y_true_val.append(y_test_copy)
            y_pred_val.append(y_pred_user)
            j += 1
        train_scores.append(train_val_scores)
        test_scores.append(test_val_scores)

        y_true.append(flat_list_of_array(y_true_val))
        y_pred.append(flat_list_of_array(y_pred_val))

    write_results_models(train_scores=train_scores, test_scores=test_scores, class_names=lab_enc.classes_,
                         y_pred=y_pred, y_true=y_true, file_path=model_path)
    print('It took ', time.time() - start, ' seconds.')

    return None


def strat_shuffle_split(x, y, split=0.3, random_state=12345):
    try:
        x_add_train, x_test, y_add_train, y_test = train_test_split(x, y, train_size=split, stratify=y)
    except ValueError:
        print("Stratified Shuffle split is not possible")
        x_add_train, x_test, y_add_train, y_test = train_test_split(x, y, train_size=split)
    return x_add_train, x_test, y_add_train, y_test


def train_clf_user_calibrated(pipe, clf_params, train_size=0.6, model_path=MODEL_PATH):
    start = time.time()
    x, y, user, lab_enc = airware_baseline_data()
    train_scores, test_scores = [], []
    y_true, y_pred = [], []

    # Delete near zero variance columns
    nz_var_ind = remove_near_zero_var(x, thresh=20)
    x = np.delete(x, nz_var_ind, axis=1)

    logo = LeaveOneGroupOut()
    i = 0
    for train_idx, test_idx in logo.split(x, y, user):
        i += 1
        print("\nUser:", i)

        x_train, y_train = x[train_idx, :], y[train_idx]
        x_test, y_test = x[test_idx, :], y[test_idx]

        train_val_scores, test_val_scores = [], []
        y_true_val, y_pred_val = [], []

        for j in range(CV_FOLDS):
            print("Fold:", j)
            seed_gen = j * 200
            # Split user test data - 60% added to the training data set
            x_add, x_test_new, y_add, y_test_new = strat_shuffle_split(x_test, y_test, split=train_size,
                                                                       random_state=seed_gen)

            # Add additional training data to the original
            x_train_new = np.vstack((x_train, x_add))
            y_train_new = np.vstack((y_train.reshape((-1,1)), y_add.reshape((-1,1))))
            y_train_new = y_train_new.reshape(-1)

            sort_idx = np.argsort(y_test_new.reshape(-1))
            x_test_new = x_test_new[sort_idx, :]
            y_test_new = y_test_new[sort_idx]

            x_train_copy = x_train_new.copy()
            y_train_copy = y_train_new.copy()

            x_test_copy = x_test_new.copy()
            y_test_copy = y_test_new.copy()

            # Call model function - Refit a new model
            clf_pipe = pipe.set_params(**clf_params)

            # Fit model
            clf_pipe.fit(x_train_copy, y_train_copy)

            # Predictions
            y_pred_user = clf_pipe.predict(x_test_copy)

            # Evaluate training scores/accuracy
            train_val_scores.append(clf_pipe.score(x_train_copy, y_train_copy))
            # Evaluate test scores/accuracy
            test_val_scores.append(clf_pipe.score(x_test_copy, y_test_copy))
            del clf_pipe, x_train_new, y_train_new, x_train_copy, y_train_copy

            if train_size == 0.6:
                y_true_val.append(y_test_copy)
                y_pred_val.append(y_pred_user)

        train_scores.append(train_val_scores)
        test_scores.append(test_val_scores)
        if train_size == 0.6:
            y_true.append(y_true_val)
            y_pred.append(y_pred_val)
    write_results_models(train_scores=train_scores, test_scores=test_scores, class_names=lab_enc.classes_,
                         y_pred=y_pred, y_true=y_true, file_path=model_path)
    print('It took ', time.time() - start, ' seconds.')

    return None
