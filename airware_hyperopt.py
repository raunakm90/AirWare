import os
import json
import argparse
from data import Read_Data
from keras.layers import Reshape, merge, concatenate
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from utils.generators import *
from keras.regularizers import l2
from keras.models import Input, Model
from keras import backend as K
from keras import optimizers

from sklearn.model_selection import train_test_split

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice, normal


def airware_data():
    param_list = HyperParams()
    gd = Read_Data.GestureData(gest_set=1)
    x, y, user, input_shape, lab_enc = gd.compile_data(nfft=param_list.NFFT_VAL, overlap=param_list.OVERLAP,
                                                       brange=param_list.BRANGE, max_seconds=2.5,
                                                       keras_format=True,
                                                       plot_spectogram=False,
                                                       baseline_format=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=None, random_state=234)
    num_classes = len(lab_enc.classes_)
    param_list.input_shape = input_shape
    param_list.num_classes = num_classes

    return x_train, x_test, y_train, y_test, param_list


def split_model_1(x_train, x_test, y_train, y_test, param_list):
    np.random.seed(234)
    l2_val = l2({{normal(param_list.L2_VAL['mu'], param_list.L2_VAL['std'])}})
    image_input = Input(
        shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], param_list.input_shape[1] - 2))(image_input)
    x = Conv1D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling1D(2)(x))

    ir_input = Input(shape=(param_list.input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], 2))(ir_input)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling1D(2)(x))

    x = concatenate([image_x, ir_x])

    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    preds = Dense(param_list.num_classes, activation='softmax',
                  kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}})(x)

    model = Model([image_input, ir_input], preds)
    rmsprop = optimizers.rmsprop(lr={{choice(param_list.LR_VAL)}})
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])

    model.fit_generator(
        create_generator([x_train[:, :, 0:-2, :], x_train[:, :, -2:, :]], y_train, batch_size=param_list.BATCH_SIZE),
        steps_per_epoch=int(len(x_train) / param_list.BATCH_SIZE),
        epochs={{choice(param_list.NB_EPOCHS)}}, verbose=0)

    score, acc = model.evaluate_generator(
        create_generator([x_test[:, :, 0:-2, :], x_test[:, :, -2:, :]], y_test, batch_size=param_list.BATCH_SIZE),
        steps=len(x_test)
    )
    print("Test Accuracy: ", acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def split_model_2(x_train, x_test, y_train, y_test, param_list):
    np.random.seed(234)
    l2_val = l2({{normal(param_list.L2_VAL['mu'], param_list.L2_VAL['std'])}})
    image_input = Input(
        shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], param_list.input_shape[1] - 2))(image_input)
    x = Conv1D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling1D(2)(x))

    ir_input = Input(shape=(param_list.input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], 2))(ir_input)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling1D(2)(x))

    x = concatenate([image_x, ir_x])

    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    x = Dropout({{uniform(param_list.DROPOUT_VAL['lower'], param_list.DROPOUT_VAL['upper'])}})(x)
    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    x = Dropout({{uniform(param_list.DROPOUT_VAL['lower'], param_list.DROPOUT_VAL['upper'])}})(x)
    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    x = Dropout({{uniform(param_list.DROPOUT_VAL['lower'], param_list.DROPOUT_VAL['upper'])}})(x)

    preds = Dense(param_list.num_classes, activation='softmax',
                  kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}})(x)

    model = Model([image_input, ir_input], preds)
    rmsprop = optimizers.rmsprop(lr={{choice(param_list.LR_VAL)}})
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    model.fit_generator(
        create_generator([x_train[:, :, 0:-2, :], x_train[:, :, -2:, :]], y_train, batch_size=param_list.BATCH_SIZE),
        steps_per_epoch=int(len(x_train) / param_list.BATCH_SIZE),
        epochs={{choice(param_list.NB_EPOCHS)}}, verbose=0)

    score, acc = model.evaluate_generator(
        create_generator([x_test[:, :, 0:-2, :], x_test[:, :, -2:, :]], y_test, batch_size=param_list.BATCH_SIZE),
        steps=len(x_test)
    )
    print("Test Accuracy: ", acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def split_model_3(x_train, x_test, y_train, y_test, param_list):
    np.random.seed(234)
    l2_val = l2({{normal(param_list.L2_VAL['mu'], param_list.L2_VAL['std'])}})
    image_input = Input(
        shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], param_list.input_shape[1] - 2))(image_input)
    x = Conv1D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling1D(2)(x))

    ir_input = Input(shape=(param_list.input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], 2))(ir_input)
    x = Conv1D(2, 2, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(2, 2, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling1D(2)(x))

    x = concatenate([image_x, ir_x])
    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    x = Dropout({{uniform(param_list.DROPOUT_VAL['lower'], param_list.DROPOUT_VAL['upper'])}})(x)
    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    x = Dropout({{uniform(param_list.DROPOUT_VAL['lower'], param_list.DROPOUT_VAL['upper'])}})(x)
    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    x = Dropout({{uniform(param_list.DROPOUT_VAL['lower'], param_list.DROPOUT_VAL['upper'])}})(x)

    preds = Dense(param_list.num_classes, activation='softmax',
                  kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}})(x)

    model = Model([image_input, ir_input], preds)
    rmsprop = optimizers.rmsprop(lr={{choice(param_list.LR_VAL)}})
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    model.fit_generator(
        create_generator([x_train[:, :, 0:-2, :], x_train[:, :, -2:, :]], y_train, batch_size=param_list.BATCH_SIZE),
        steps_per_epoch=int(len(x_train) / param_list.BATCH_SIZE),
        epochs={{choice(param_list.NB_EPOCHS)}}, verbose=0)

    score, acc = model.evaluate_generator(
        create_generator([x_test[:, :, 0:-2, :], x_test[:, :, -2:, :]], y_test, batch_size=param_list.BATCH_SIZE),
        steps=len(x_test)
    )
    print("Test Accuracy: ", acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def split_model_4(x_train, x_test, y_train, y_test, param_list):
    np.random.seed(234)
    l2_val = l2({{normal(param_list.L2_VAL['mu'], param_list.L2_VAL['std'])}})
    image_input = Input(
        shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1))(image_input)
    x = Conv2D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D({{choice(param_list.IMG_CONV_FILTERS)}}, {{choice(param_list.IMG_CONV_SIZE)}}, padding='same',
               activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling2D(2)(x))

    ir_input = Input(shape=(param_list.input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], 2, 1))(ir_input)
    x = Conv2D(2, 3, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(2, 3, padding='same', activation='relu',
               kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}}, kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling2D(2)(x))

    x = concatenate([image_x, ir_x])

    x = Dense({{choice(param_list.HIDDEN_UNITS)}}, activation='relu',
              kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}},
              kernel_regularizer=l2_val)(x)
    preds = Dense(param_list.num_classes, activation='softmax',
                  kernel_initializer={{choice(param_list.KERNEL_INITIALIZER)}})(x)

    model = Model([image_input, ir_input], preds)
    rmsprop = optimizers.rmsprop(lr={{choice(param_list.LR_VAL)}})
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])

    model.fit_generator(
        create_generator([x_train[:, :, 0:-2, :], x_train[:, :, -2:, :]], y_train, batch_size=param_list.BATCH_SIZE),
        steps_per_epoch=int(len(x_train) / param_list.BATCH_SIZE),
        epochs={{choice(param_list.NB_EPOCHS)}}, verbose=0)

    score, acc = model.evaluate_generator(
        create_generator([x_test[:, :, 0:-2, :], x_test[:, :, -2:, :]], y_test, batch_size=param_list.BATCH_SIZE),
        steps=len(x_test)
    )
    print("Test Accuracy: ", acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def hyperparam_search(model_fn, file_path):
    K.clear_session()
    x_train, x_test, y_train, y_test, param_list = airware_data()
    functions = [create_generator, HyperParams]
    best_run, best_model = optim.minimize(model=model_fn,
                                          data=airware_data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials(),
                                          functions=functions)

    print("Evalutation of best performing model:")
    print(best_model.evaluate([x_test[:, :, 0:-2, :], x_test[:, :, -2:, :]], y_test))
    print("Best Parameters")
    print(best_run)
    with open(file_path + 'model_best_run.json', 'w') as fp:
        json.dump(best_run, fp)


def hyper_opt_split_model_1():
    f_path = "./gridSearch/split_model_1/"
    if not os.path.exists(f_path):
        os.makedirs(f_path)
        hyperparam_search(split_model_1, f_path)


def hyper_opt_split_model_2():
    f_path = "./gridSearch/split_model_2/"
    if not os.path.exists(f_path):
        os.makedirs(f_path)
        hyperparam_search(split_model_2, f_path)


def hyper_opt_split_model_3():
    f_path = "./gridSearch/split_model_3/"
    if not os.path.exists(f_path):
        os.makedirs(f_path)
        hyperparam_search(split_model_3, f_path)


def hyper_opt_split_model_4():
    f_path = "./gridSearch/split_model_4/"
    if not os.path.exists(f_path):
        os.makedirs(f_path)
        hyperparam_search(split_model_4, f_path)


if __name__ == '__main__':
    function_map = {'model_1': hyper_opt_split_model_1,
                    'model_2': hyper_opt_split_model_2,
                    'model_3': hyper_opt_split_model_3,
                    'model_4': hyper_opt_split_model_4}
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument('-model',
                        help="Select model",
                        choices=['model_1', 'model_2',
                                 'model_3', 'model_4'])
    args = parser.parse_args()
    function = function_map[args.model]
    print("Model:", args.model)
    function()
