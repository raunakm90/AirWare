import numpy as np
from .generators import create_generator
from .generate_report import write_results_models, write_train_hist
import utils.models as model
from data import Read_Data
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit, train_test_split
import json
import keras.backend as K
import keras


def airware_data(gest_set=1):
    param_list = model.HyperParams()
    gd = Read_Data.GestureData(gest_set=gest_set)
    x, y, user, input_shape, lab_enc = gd.compile_data(nfft=param_list.NFFT_VAL, overlap=param_list.OVERLAP,
                                                       brange=param_list.BRANGE, max_seconds=2.5,
                                                       keras_format=True,
                                                       plot_spectogram=False,
                                                       baseline_format=False)
    num_classes = len(lab_enc.classes_)
    param_list.input_shape = input_shape
    param_list.num_classes = num_classes
    return x, y, user, lab_enc, param_list


# CV Strategy: Leave one subject out
def loso_cv(model_fn, gest_set, hyper_param_path, results_file_path):
    x, y, user, lab_enc, param_list = airware_data(gest_set)

    # Read the best parameters for the given model
    with open(hyper_param_path + "model_best_run.json") as fname:
        hyper_param_dict = json.load(fname)

    logo = LeaveOneGroupOut()
    train_scores, test_scores = [], []
    train_val_hist = []
    y_true, y_hat = [], []
    i = 1

    # Callback for early stopping. Patience monitor number of epochs. Min_Delta monitors the change in metric.
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=10e-5, verbose=1)
    # Leave One subject out
    for train_idx, test_idx in logo.split(x, y, user):
        K.clear_session()
        print("\nUser:", i)
        i += 1
        # Train and test data - leave one subject out
        x_train, y_train = x[train_idx, :, :, :], y[train_idx]
        x_test, y_test = x[test_idx, :, :, :], y[test_idx]

        # Create copies of the train and test data sets
        x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
        x_test_copy, y_test_copy = x_test.copy(), y_test.copy()

        # Call model function
        split_model = model_fn(hyper_param_dict, param_list)

        # steps_per_epoch = how many generators to go through per epoch
        train_val_hist.append(split_model.fit_generator(
            create_generator([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy,
                             batch_size=param_list.BATCH_SIZE),
            steps_per_epoch=int(len(x_train_copy) / param_list.BATCH_SIZE),
            epochs=param_list.NB_EPOCHS[hyper_param_dict['epochs']], verbose=0,
            validation_data=([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy)))
        # Evaluate training scores
        train_scores.append(
            split_model.evaluate([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy))
        # Evaluate test scores
        test_scores.append(
            split_model.evaluate([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy))

        # Predict for test data
        yhat = np.argmax(split_model.predict([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]]), axis=1)
        y_hat.append(yhat)
        y_true.append(y_test_copy)

    write_results_models(train_scores, test_scores, lab_enc.classes_, y_hat, y_true, results_file_path)
    write_train_hist(train_val_hist, results_file_path)

    return None


# CV Strategy: Personalized
def personalized_cv(model_fn, gest_set, hyper_param_path, results_file_path):
    x, y, user, lab_enc, param_list = airware_data(gest_set)

    # Read the best parameters for the given model
    with open(hyper_param_path + "model_best_run.json") as fname:
        hyper_param_dict = json.load(fname)

    # CV object
    logo = LeaveOneGroupOut()

    y_hat_user, y_true_user = [], []
    test_scores_user, train_scores_user = [], []
    train_hist_user = []
    i = 0  # Keep track of users

    for rem_idx, keep_idx in logo.split(x, y, user):
        K.clear_session()
        print("\nUser:", i)
        i += 1
        user_name = "User_" + str(i)

        # Keep only the user data
        # x_rem, y_rem = x[rem_idx,:,:,:], y[rem_idx]
        x_keep, y_keep = x[keep_idx, :, :, :], y[keep_idx]
        train_scores, test_scores = [], []
        train_val_hist = []
        y_true, y_hat = [], []

        # Define CV object
        cv_strat = StratifiedShuffleSplit(n_splits=param_list.cv_folds, test_size=0.4, random_state=i * 243)
        j = 0  # Keep track of CV Fold

        # Stratified cross validation for each user
        for train_idx, test_idx in cv_strat.split(x_keep, y_keep):
            print('Fold:', str(j))
            x_train, y_train = x_keep[train_idx, :, :, :], y_keep[train_idx]
            x_test, y_test = x_keep[test_idx, :, :, :], y_keep[test_idx]

            x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
            x_test_copy, y_test_copy = x_test.copy(), y_test.copy()

            split_model = model_fn(hyper_param_dict, param_list)
            train_val_hist.append(split_model.fit_generator(
                create_generator([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy,
                                 batch_size=param_list.BATCH_SIZE),
                steps_per_epoch=int(len(x_train_copy) / param_list.BATCH_SIZE),
                # how many generators to go through per epoch
                epochs=param_list.NB_EPOCHS[hyper_param_dict['epochs']], verbose=0,
                validation_data=([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy)))

            train_scores.append(
                split_model.evaluate([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy))
            test_scores.append(
                split_model.evaluate([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy))

            y_hat.append(
                np.argmax(split_model.predict([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]]), axis=1))
            y_true.append(y_test)
            j += 1

        y_hat_user.append(y_hat)
        y_true_user.append(y_true)
        train_scores_user.append(train_scores)
        test_scores_user.append(test_scores)
        write_train_hist(train_val_hist, results_file_path + user_name)

    write_results_models(train_scores_user, test_scores_user, lab_enc.classes_, y_hat_user, y_true_user,
                               results_file_path)
    return None


def strat_shuffle_split(x, y, split=0.3, random_state=12345):
    try:
        x_add_train, x_test, y_add_train, y_test = train_test_split(x, y, train_size=split, stratify=y)
    except ValueError:
        print("Stratified Shuffle split is not possible")
        x_add_train, x_test, y_add_train, y_test = train_test_split(x, y, train_size=split)
    return x_add_train, x_test, y_add_train, y_test


# 60-40 User split CV
def user_split_cv(model_fn, gest_set, hyper_param_path, results_file_path):
    x, y, user, lab_enc, param_list = airware_data(gest_set)

    # Read the best parameters for the given model
    with open(hyper_param_path + "model_best_run.json") as fname:
        hyper_param_dict = json.load(fname)

    logo = LeaveOneGroupOut()
    train_score, test_score = [], []
    y_hat, y_true = [], []
    class_names = []
    train_val_hist = []
    i = 0
    for train_idx, test_idx in logo.split(x, y, user):
        K.clear_session()
        i += 1
        user_name = "User_" + str(i)
        print("\nUser:", i)

        x_train, y_train = x[train_idx, :, :, :], y[train_idx]
        x_test, y_test = x[test_idx, :, :, :], y[test_idx]

        cv_train_score, cv_test_score = [], []
        cv_yhat, cv_ytrue = [], []
        cv_train_val_hist = []

        for j in range(param_list.cv_folds):
            print("Fold:", j)
            seed_gen = j * 200
            # Split user test data - 60% added to the training data set
            x_add, x_test_new, y_add, y_test_new = strat_shuffle_split(x_test, y_test, split=0.6, random_state=seed_gen)

            # Add additional training data to the original
            x_train = np.vstack((x_train, x_add))
            y_train = np.vstack((y_train, y_add))

            sort_idx = np.argsort(y_test_new.reshape(-1))
            x_test_new = x_test_new[sort_idx, :, :, :]
            y_test_new = y_test_new[sort_idx]

            x_train_copy = x_train.copy()
            y_train_copy = y_train.copy()

            x_test_copy = x_test_new.copy()
            y_test_copy = y_test_new.copy()

            split_model = model_fn(hyper_param_dict, param_list)
            cv_train_val_hist.append(split_model.fit_generator(
                create_generator([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy,
                                 batch_size=param_list.BATCH_SIZE),
                steps_per_epoch=int(len(x_train_copy) / param_list.BATCH_SIZE),
                # how many generators to go through per epoch
                epochs=param_list.NB_EPOCHS[hyper_param_dict['epochs']], verbose=0,
                validation_data=([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy)))

            cv_train_score.append(
                split_model.evaluate([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy))
            cv_test_score.append(
                split_model.evaluate([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy))

            y_hat_temp = np.argmax(split_model.predict([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]]), axis=1)
            cv_ytrue.append(y_test_copy)
            cv_yhat.append(y_hat_temp)

        train_score.append(cv_train_score)
        test_score.append(cv_test_score)
        y_hat.append(cv_yhat)
        y_true.append(cv_ytrue)
        write_train_hist(train_val_hist, results_file_path + user_name)
    write_results_models(train_score, test_score, lab_enc.classes_, y_hat, y_true,
                               results_file_path)
    return None