from data import Read_Data
import numpy as np
from keras.layers import Reshape, merge, concatenate
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Input, Model
from keras import backend as K

from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
import sklearn.metrics as mt

import matplotlib.pyplot as plt
import argparse

NFFT_VAL = 4096
OVERLAP = 0.5
BRANGE = 16


# @TODO: Add args to grid search model parameters
def split_model_1():
    l2_val = l2(0.001)
    image_input = Input(
        shape=(input_shape[0], input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(input_shape[0], input_shape[1] - 2))(image_input)
    # Convolution Layer 1
    x = Conv1D(8, 3, padding='same', activation='relu',
               kernel_initializer='he_uniform', kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    # Convolution Layer 2
    x = Conv1D(16, 3, padding='same', activation='relu',
               kernel_initializer='he_uniform', kernel_regularizer=l2_val)(x)
    # x = Conv1D(1, 3, padding='same', activation='relu',kernel_initializer='he_uniform',kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling1D(2)(x))

    ir_input = Input(shape=(input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(input_shape[0], 2))(ir_input)
    # Convolution Layer 1
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer='he_uniform', kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    # Convolution Layer 2
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer='he_uniform', kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling1D(2)(x))

    x = concatenate([image_x, ir_x])
    # x = Flatten()(x)
    # Dense Network - MLP
    x = Dense(100, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l2_val)(x)
    preds = Dense(NUM_CLASSES, activation='softmax',
                  kernel_initializer='glorot_uniform')(x)

    model = Model([image_input, ir_input], preds)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model


def create_generator(XI, Y, batch_size=64):
    X = XI[0]
    I = XI[1]
    while True:
        # shuffled indices
        idx = np.random.permutation(X.shape[0])
        # create image generator
        datagenSpec = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # 180,  # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=0,
            # 0.1,  # randomly shift images horizontally (fraction of total
            # width)
            width_shift_range=0.0,
            # 0.1,  # randomly shift images vertically (fraction of total
            # height)
            height_shift_range=0.1,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        batchesSpec = datagenSpec.flow(
            X[idx], Y[idx], batch_size=batch_size, shuffle=False)

        # create image generator
        datagenIR = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # 180,  # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=0,
            # 0.1,  # randomly shift images horizontally (fraction of total
            # width)
            width_shift_range=0.0,
            # 0.1,  # randomly shift images vertically (fraction of total
            # height)
            height_shift_range=0.1,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        batchesIR = datagenIR.flow(
            I[idx], Y[idx], batch_size=batch_size, shuffle=False)

        for b1, b2 in zip(batchesSpec, batchesIR):
            # print(b1[0].shape,b2[0].shape)
            yield [b1[0], b2[0]], b1[1]

        break


def write_results(train_scores, test_scores, class_names, y_hat, y_true, file_path, train_val_hist):
    print("Writing results...")
    np.savez(file_path + "_Train_Scores.npz", train_scores)
    np.savez(file_path + "_Test_Scores.npz", test_scores)
    np.savez(file_path + "_Predictions.npz", y_hat)
    np.savez(file_path + "_Truth.npz", y_true)
    np.savez(file_path + "_Class_Names.npz", class_names)
    np.savez(file_path + "_Train_Val_Hist.npz", train_val_hist)


# Leave one subject out CV
def loso_cv(cv_folds, gest_set, nb_epoch):
    global NUM_CLASSES, input_shape
    file_path = "./leave_one_subject" + "/gest_set_" + str(gest_set) + "/FinalModel"
    batch_size = 10

    gd = Read_Data.GestureData(gest_set=gest_set)
    print("Reading data")
    x, y, user, input_shape, lab_enc = gd.compile_data(nfft=NFFT_VAL, overlap=OVERLAP,
                                                       brange=BRANGE, keras_format=True,
                                                       plot_spectogram=False,
                                                       baseline_format=False)
    NUM_CLASSES = len(lab_enc.classes_)
    logo = LeaveOneGroupOut()

    train_scores, test_scores = [], []
    train_val_hist = []
    y_true, y_hat = [], []
    class_names = []
    i = 1

    for train_idx, test_idx in logo.split(x, y, user):
        print("\nUser:", i)
        i += 1
        # Train and test data - leave one subject out
        x_train, y_train = x[train_idx, :, :, :], y[train_idx]
        x_test, y_test = x[test_idx, :, :, :], y[test_idx]

        # Create copies of the train and test data sets
        x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
        x_test_copy, y_test_copy = x_test.copy(), y_test.copy()

        # Call model function
        split_model = split_model_1()

        # steps_per_epoch = how many generators to go through per epoch
        train_val_hist.append(split_model.fit_generator(
            create_generator([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy,
                             batch_size=batch_size), steps_per_epoch=int(len(x_train_copy) / batch_size),
            epochs=nb_epoch, verbose=0,
            validation_data=([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy)))
        # Evaluate training scores
        train_scores.append(
            split_model.evaluate([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy))
        # Evaluate test scores
        test_scores.append(
            split_model.evaluate([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy))

        # Predict for test data
        yhat = np.argmax(split_model.predict([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]]), axis=1)
        class_names.append(lab_enc.classes_[y_test_copy])
        y_hat.append(yhat)
        y_true.append(y_test_copy)
        K.clear_session()
    write_results(train_scores, test_scores, class_names, y_hat, y_true, file_path,
                  train_val_hist)
    return train_val_hist


def strat_shuffle_split(x, y, split=0.3, random_state=12345):
    cv_obj = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=random_state)
    for train_idx, test_idx in cv_obj.split(x, y):
        x_add_train, y_add_train = x[train_idx, :, :, :], y[train_idx]
        x_test, y_test = x[test_idx, :, :, :], y[test_idx]

    return x_add_train, x_test, y_add_train, y_test


# 60-40 User split CV
def user_split_cv(cv_folds, nb_epoch, gest_set):
    global NUM_CLASSES, input_shape
    file_path = "./user_split_cv" + "/gest_set_" + str(gest_set) + "/FinalModel"
    batch_size = 10

    gd = Read_Data.GestureData(gest_set=gest_set)
    print("Reading data")
    x, y, user, input_shape, lab_enc = gd.compile_data(nfft=NFFT_VAL, overlap=OVERLAP,
                                                       brange=BRANGE, keras_format=True,
                                                       plot_spectogram=False,
                                                       baseline_format=False)
    NUM_CLASSES = len(lab_enc.classes_)

    logo = LeaveOneGroupOut()
    train_score, test_score = [], []
    y_hat, y_true = [], []
    class_names = []
    train_val_hist = []
    i = 0
    for train_idx, test_idx in logo.split(x, y, user):
        i += 1
        print("\nUser:", i)

        x_train, y_train = x[train_idx, :, :, :], y[train_idx]
        x_test, y_test = x[test_idx, :, :, :], y[test_idx]

        cv_train_score, cv_test_score = [], []
        cv_yhat, cv_ytrue = [], []
        cv_class_names = []
        cv_train_val_hist = []

        for j in range(cv_folds):
            print("Fold:", j)
            seed_gen = j * 200
            # Split user test data - 60% added to the training data set
            x_add, x_test_new, y_add, y_test_new = strat_shuffle_split(x_test, y_test, split=0.4, random_state=seed_gen)

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

            split_model = split_model_1()
            cv_train_val_hist.append(split_model.fit_generator(
                create_generator([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy,
                                 batch_size=batch_size),
                steps_per_epoch=int(len(x_train_copy) / batch_size),  # how many generators to go through per epoch
                epochs=nb_epoch, verbose=0,
                validation_data=([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy)))

            cv_train_score.append(
                split_model.evaluate([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy))
            cv_test_score.append(
                split_model.evaluate([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy))

            y_hat_temp = np.argmax(split_model.predict([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]]), axis=1)
            cv_class_names.append(lab_enc.classes_[y_test_copy])
            cv_ytrue.append(y_test_copy)
            cv_yhat.append(y_hat_temp)

        train_score.append(cv_train_score)
        test_score.append(cv_test_score)
        y_hat.append(cv_yhat)
        y_true.append(cv_ytrue)
        class_names.append(cv_class_names)
        train_val_hist.append(cv_train_val_hist)
        K.clear_session()
    write_results(train_score, test_score, class_names, y_hat, y_true, file_path=file_path)


# Personalized user CV
def personalized_cv(cv_folds, nb_epoch, gest_set):
    global NUM_CLASSES, input_shape
    file_path = "./personalized_cv" + "/gest_set_" + str(gest_set) + "/FinalModel"
    batch_size = 10
    # CV object
    logo = LeaveOneGroupOut()
    i = 0

    gd = Read_Data.GestureData(gest_set=gest_set)
    print("Reading data")
    x, y, user, input_shape, lab_enc = gd.compile_data(nfft=NFFT_VAL, overlap=OVERLAP,
                                                       brange=BRANGEf, keras_format=True,
                                                       plot_spectogram=False,
                                                       baseline_format=False)
    NUM_CLASSES = len(lab_enc.classes_)

    y_hat_user, y_test_user = [], []
    class_names_user = []
    test_scores_user, train_scores_user = [], []
    train_hist_user = []

    for rem_idx, keep_idx in logo.split(x, y, user):
        print("\nUser:", i)
        i += 1
        user_name = "User_" + str(i)

        # Keep only the user data
        # x_rem, y_rem = x[rem_idx,:,:,:], y[rem_idx]
        x_keep, y_keep = x[keep_idx, :, :, :], y[keep_idx]
        train_scores, test_scores = [], []
        train_val_hist = []
        y_true, y_hat = [], []
        class_names = []

        # Define CV object
        cv_strat = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.4, random_state=i * 12345)
        j = 0

        # Stratified cross validation for each user
        for train_idx, test_idx in cv_strat.split(x_keep, y_keep):
            print('Fold:', str(j))
            x_train, y_train = x_keep[train_idx, :, :, :], y_keep[train_idx]
            x_test, y_test = x_keep[test_idx, :, :, :], y_keep[test_idx]

            x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
            x_test_copy, y_test_copy = x_test.copy(), y_test.copy()

            split_model = split_model_1()
            train_val_hist.append(split_model.fit_generator(
                create_generator([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy,
                                 batch_size=batch_size),
                steps_per_epoch=int(len(x_train_copy) / batch_size),  # how many generators to go through per epoch
                epochs=nb_epoch, verbose=0,
                validation_data=([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy)))

            train_scores.append(
                split_model.evaluate([x_train_copy[:, :, 0:-2, :], x_train_copy[:, :, -2:, :]], y_train_copy))
            test_scores.append(
                split_model.evaluate([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]], y_test_copy))

            y_hat.append(
                np.argmax(split_model.predict([x_test_copy[:, :, 0:-2, :], x_test_copy[:, :, -2:, :]]), axis=1))
            y_true.append(y_test)
            class_names.append(lab_enc.classes_[y_test_copy])
            j += 1

        K.clear_session()
        y_hat_user.append(y_hat)
        y_test_user.append(y_true)
        class_names_user.append(class_names)
        train_scores_user.append(train_scores)
        test_scores_user.append(test_scores)
        train_hist_user.append(train_val_hist)

    write_results(train_scores_user, test_scores_user, class_names_user, y_hat_user, y_test_user, file_path,
                  train_val_hist)
    return train_hist_user


if __name__ == '__main__':
    function_map = {'loso': loso_cv,
                    'user_split': user_split_cv,
                    'personalized_cv': personalized_cv}
    parser = argparse.ArgumentParser(description="AirWare grid search and train model using different CV strategies")
    # "?" one argument consumed from the command line and produced as a single item
    # Positional arguments
    parser.add_argument('-cv_strategy',
                        help="Define CV Strategy. loso: Leave one subject out, user_split: Partial train and test "
                             "user, personalized_cv: Train and test only for a given user",
                        choices=['loso', 'user_split',
                                 'personalized_cv'])
    parser.add_argument('-gesture_set', type=int, default=1,
                        help="Gesture set. 1: All gestures, 2: Reduced Gesture 1, 3: Reduced Gesture 2, 4: Reduced "
                             "Gesture 3, 5: Reduced Gesture 4",
                        choices=range(1, 6))
    parser.add_argument('-cv_folds', type=int, help="Number of Cross validation folds", default=5)
    parser.add_argument('-nb_epoch', type=int, help="Number of epochs that trains the model", default=10)
    args = parser.parse_args()
    function = function_map[args.cv_strategy]
    print("Cross Validation Strategy:", function)
    function(args.cv_folds, args.gesture_set, args.nb_epoch)
