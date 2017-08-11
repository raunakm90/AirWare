import numpy as np
from keras.preprocessing.image import ImageDataGenerator


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
        self.KERNEL_INITIALIZER = ['he_uniform', 'glorot_uniform', 'lecun_uniform',
                                   'he_normal', 'glorot_normal', 'lecun_normal']
        self.cv_folds = 5

    def __str__(self):
        print("Class to define and store optimization and model parameters")


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
            width_shift_range=0.1,
            # 0.1,  # randomly shift images vertically (fraction of total
            # height)
            height_shift_range=0.1,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,
            fill_mode='nearest')  # randomly flip images

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
            width_shift_range=0.1,
            # 0.1,  # randomly shift images vertically (fraction of total
            # height)
            height_shift_range=0.0,
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,
            fill_mode='nearest')  # randomly flip images

        batchesIR = datagenIR.flow(
            I[idx], Y[idx], batch_size=batch_size, shuffle=False)

        for b1, b2 in zip(batchesSpec, batchesIR):
            # print(b1[0].shape,b2[0].shape)
            yield [b1[0], b2[0]], b1[1]

        break
