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
        # suffled indices
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


def write_results(train_scores, test_scores, class_names, y_hat, y_true, file_path):
    print("Writing results...")
    np.savez(file_path + "_Train_Scores.npz", train_scores)
    np.savez(file_path + "_Test_Scores.npz", test_scores)
    np.savez(file_path + "_Predictions.npz", y_hat)
    np.savez(file_path + "_Truth.npz", y_true)


def plot_train_hist(train_val_hist, file_path):
    fig, axarr = plt.subplots(1, 2, sharey=True)
    i = 1
    for item in train_val_hist:
        axarr[0].plot(item.history['loss'], label='User_' + str(i))
        axarr[1].plot(item.history['val_loss'], label='User_' + str(i))
        i += 1
    plt.legend(loc='best')
    plt.savefig(file_path + "Loss_History.png")


# Leave one subject out CV
def loso_gridSearch_cv(x, y, user, lab_enc, batch_size=10, nb_epoch=10, file_path="./gridSearch/Exp"):
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
        # K.clear_session()
    write_results(train_scores, test_scores, class_names, y_hat, y_true, file_path)
    return train_val_hist


def grid_search(nb_epoch=200):
    global NUM_CLASSES, input_shape
    file_path = "./gridSearch/"
    nfft_try = [int(4096), int(2048), int(1024)]
    overlap_try = [0.9, 0.5, 0.75]
    brange_try = [8, 16]

    K.clear_session()
    # Grid Search
    for nfft_val in nfft_try:
        for overlap_val in overlap_try:
            for brange_val in brange_try:
                # Define file name to store results
                fname = file_path + "Exp_" + str(nfft_val) + "_" + str(overlap_val) + "_" + str(brange_val)
                # Read and format data - all gestures
                gd = Read_Data.GestureData(gest_set=1)
                print("Reading data")
                x, y, user, input_shape, lab_enc = gd.compile_data(nfft=nfft_val, overlap=overlap_val,
                                                                   brange=brange_val, keras_format=True,
                                                                   plot_spectogram=False,
                                                                   baseline_format=False)
                NUM_CLASSES = len(lab_enc.classes_)
                print("NFFT_Val: ", nfft_val, "Overlap_Val: ", overlap_val, "Brange_Val:", brange_val)
                print("Train the model")
                train_val_hist = loso_gridSearch_cv(x, y, user,
                                                    lab_enc,
                                                    nb_epoch=nb_epoch,
                                                    file_path=fname)
                plot_train_hist(train_val_hist, file_path=fname)
                K.clear_session()


if __name__ == '__main__':
    print("Grid Search FFT parameters")
    grid_search()
