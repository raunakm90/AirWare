from keras.layers import Reshape, concatenate
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from utils.generators import *
from keras.regularizers import l2
from keras.models import Input, Model
from keras import optimizers
import numpy as np

np.random.seed(384)


class HyperParams():
    def __init__(self):
        self.NFFT_VAL = 4096
        self.BRANGE = 16
        self.OVERLAP = 0.5
        self.LR_VAL = [10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1]
        self.NB_EPOCHS = [100, 200, 300, 400]
        self.L2_VAL = {'mu': 0.001, 'std': 0.0001}
        self.DROPOUT_VAL = {'upper': 1, 'lower': 0}
        self.HIDDEN_UNITS = [512, 256, 128, 64, 32]
        self.IMG_CONV_FILTERS = [8, 16, 32, 64]
        self.IMG_CONV_SIZE = [2, 3, 5]
        self.BATCH_SIZE = 32
        self.KERNEL_INITIALIZER = ['he_uniform', 'glorot_uniform', 'lecun_uniform']
        self.cv_folds = 5

    def __str__(self):
        print("Class to define and store optimization and model parameters")


def split_model_1(hyper_param_dict, param_list):
    np.random.seed(234)
    l2_val = l2(hyper_param_dict['l2'])
    image_input = Input(
        shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], param_list.input_shape[1] - 2))(image_input)
    x = Conv1D(param_list.IMG_CONV_FILTERS[hyper_param_dict['Conv1D']],
               param_list.IMG_CONV_SIZE[hyper_param_dict['Conv1D_1']], padding='same',
               activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer']],
               kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(param_list.IMG_CONV_FILTERS[hyper_param_dict['Conv1D_2']],
               param_list.IMG_CONV_SIZE[hyper_param_dict['Conv1D_3']], padding='same',
               activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_1']],
               kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling1D(2)(x))

    ir_input = Input(shape=(param_list.input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], 2))(ir_input)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_2']],
               kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_3']],
               kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling1D(2)(x))

    x = concatenate([image_x, ir_x])

    x = Dense(param_list.HIDDEN_UNITS[hyper_param_dict['Dense']], activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=l2_val)(x)
    preds = Dense(param_list.num_classes, activation='softmax',
                  kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_4']])(x)

    model = Model([image_input, ir_input], preds)
    rmsprop = optimizers.rmsprop(lr=param_list.LR_VAL[hyper_param_dict['lr']])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model


def split_model_2(hyper_param_dict, param_list):
    np.random.seed(234)
    l2_val = l2(hyper_param_dict['l2'])
    image_input = Input(
        shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], param_list.input_shape[1] - 2))(image_input)
    x = Conv1D(param_list.IMG_CONV_FILTERS[hyper_param_dict['Conv1D']],
               param_list.IMG_CONV_SIZE[hyper_param_dict['Conv1D_1']], padding='same',
               activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer']],
               kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(param_list.IMG_CONV_FILTERS[hyper_param_dict['Conv1D_2']],
               param_list.IMG_CONV_SIZE[hyper_param_dict['Conv1D_3']], padding='same',
               activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_1']],
               kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling1D(2)(x))

    ir_input = Input(shape=(param_list.input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], 2))(ir_input)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_2']],
               kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(2, 3, padding='same', activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_3']],
               kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling1D(2)(x))

    x = concatenate([image_x, ir_x])

    x = Dense(param_list.HIDDEN_UNITS[hyper_param_dict['Dense']], activation='relu',
              kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_4']],
              kernel_regularizer=l2_val)(x)
    x = Dropout(hyper_param_dict['Dropout'])(x)
    x = Dense(param_list.HIDDEN_UNITS[hyper_param_dict['Dense_1']], activation='relu',
              kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_5']],
              kernel_regularizer=l2_val)(x)
    x = Dropout(hyper_param_dict['Dropout_1'])(x)
    x = Dense(param_list.HIDDEN_UNITS[hyper_param_dict['Dense_2']], activation='relu',
              kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_6']],
              kernel_regularizer=l2_val)(x)
    x = Dropout(hyper_param_dict['Dropout_2'])(x)

    preds = Dense(param_list.num_classes, activation='softmax',
                  kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_7']])(x)

    model = Model([image_input, ir_input], preds)
    rmsprop = optimizers.rmsprop(lr=param_list.LR_VAL[hyper_param_dict['lr']])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    return model


def split_model_3(hyper_param_dict, param_list):
    np.random.seed(234)
    l2_val = l2(hyper_param_dict['l2'])
    image_input = Input(
        shape=(param_list.input_shape[0], param_list.input_shape[1] - 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], param_list.input_shape[1] - 2))(image_input)
    x = Conv1D(param_list.IMG_CONV_FILTERS[hyper_param_dict['Conv1D']],
               param_list.IMG_CONV_SIZE[hyper_param_dict['Conv1D_1']], padding='same',
               activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer']],
               kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(param_list.IMG_CONV_FILTERS[hyper_param_dict['Conv1D_2']],
               param_list.IMG_CONV_SIZE[hyper_param_dict['Conv1D_3']], padding='same',
               activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_1']],
               kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(param_list.IMG_CONV_FILTERS[hyper_param_dict['Conv1D_4']],
               param_list.IMG_CONV_SIZE[hyper_param_dict['Conv1D_5']], padding='same',
               activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_2']],
               kernel_regularizer=l2_val)(x)
    image_x = Flatten()(MaxPooling1D(2)(x))

    ir_input = Input(shape=(param_list.input_shape[0], 2, 1), dtype='float32')
    x = Reshape(target_shape=(param_list.input_shape[0], 2))(ir_input)
    x = Conv1D(2, 2, padding='same', activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_3']],
               kernel_regularizer=l2_val)(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(2, 2, padding='same', activation='relu',
               kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_4']],
               kernel_regularizer=l2_val)(x)
    ir_x = Flatten()(MaxPooling1D(2)(x))

    x = concatenate([image_x, ir_x])
    x = Dense(param_list.HIDDEN_UNITS[hyper_param_dict['Dense']], activation='relu',
              kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_4']],
              kernel_regularizer=l2_val)(x)
    x = Dropout(hyper_param_dict['Dropout'])(x)
    x = Dense(param_list.HIDDEN_UNITS[hyper_param_dict['Dense_1']], activation='relu',
              kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_5']],
              kernel_regularizer=l2_val)(x)
    x = Dropout(hyper_param_dict['Dropout_1'])(x)
    x = Dense(param_list.HIDDEN_UNITS[hyper_param_dict['Dense_2']], activation='relu',
              kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_6']],
              kernel_regularizer=l2_val)(x)
    x = Dropout(hyper_param_dict['Dropout_2'])(x)

    preds = Dense(param_list.num_classes, activation='softmax',
                  kernel_initializer=param_list.KERNEL_INITIALIZER[hyper_param_dict['kernel_initializer_8']])(x)

    model = Model([image_input, ir_input], preds)
    rmsprop = optimizers.rmsprop(lr=hyper_param_dict['lr'])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model
