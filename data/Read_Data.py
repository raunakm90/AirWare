from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import signal


class Gesture:
    def __init__(self, filename, load=True, nfft=4096, overlap=0.9, brange=8, max_seconds=3):
        self.file = filename
        self.nfft = nfft
        self.overlap = int(self.nfft * overlap)
        self.brange = brange
        self.max_seconds = max_seconds
        if load:
            self.load()

    def load(self):
        with open(self.file, "r") as fid:
            # create the datatypes
            longdtype = np.dtype('>i8')
            intdtype = np.dtype('>i4')
            shortdtype = np.dtype('>i2')
            bytedtype = np.dtype('>i1')

            # load header
            self.revision = np.fromfile(fid, dtype=bytedtype, count=1)[0]
            self.sampleRate = np.fromfile(fid, dtype=intdtype, count=1)[0]
            self.freq1 = np.fromfile(fid, dtype=intdtype, count=1)[0]
            self.freq2 = np.fromfile(fid, dtype=intdtype, count=1)[0]
            self.numSamples = np.fromfile(fid, dtype=intdtype, count=1)[0]
            self.numDirSamples = np.fromfile(fid, dtype=bytedtype, count=1)[0]
            self.duration = np.fromfile(fid, dtype=longdtype, count=1)[0]

            # get data
            self.rawData = np.fromfile(
                fid, dtype=shortdtype, count=self.numSamples) / 32767.0

            # get footer data (gestures)
            gesturedtype = np.dtype(">i4, >i4, >i4")
            self.sgestures = np.fromfile(fid, dtype=gesturedtype)

            self.dirs = np.zeros(4)
            for x in range(self.numDirSamples):
                angle = self.sgestures[x][2]
                if angle >= 45 and angle < 135:
                    self.dirs[3] += 1.0
                elif angle >= 135 and angle < 225:
                    self.dirs[0] += 1.0
                elif angle >= 225 and angle < 315:
                    self.dirs[2] += 1.0
                else:
                    self.dirs[1] += 1.0

            self.loaded = True

    def __str__(self):
        ret_str = ""
        if self.loaded:
            ret_str += "Header Data\n"
            ret_str += "Revision:    v%d\n" % (self.revision)
            ret_str += "Sample rate: %d hz\n" % (self.sampleRate)
            ret_str += "Frequency 1: %d hz\n" % (self.freq1)
            ret_str += "Frequency 2: %d hz\n" % (self.freq2)
            ret_str += "Num Samples: %d\n" % (self.numSamples)
            ret_str += "Dir Samples: %d\n" % (self.numDirSamples)
            ret_str += "Duration:    %d ns\n" % (self.duration)
            ret_str += "Gestures (%d)\n" % (self.numDirSamples)
            for i in range(self.numDirSamples):
                ret_str += "G%d\n" % (i)
                ret_str += "\tIndex: %d\n" % (self.sgestures[i][0])
                ret_str += "\tSpeed: %d\n" % (self.sgestures[i][1])
                ret_str += "\tAngle: %d\n" % (self.sgestures[i][2])
        else:
            ret_str += "Not loaded yet"

        return ret_str

    def plotSpecgram(self):
        plt.figure(num=self.file)
        plt.specgram(self.rawData, NFFT=self.nfft, noverlap=self.overlap)

    def generateSpecgram(self):
        freqs, bins, Sxx = signal.spectrogram(self.rawData, nfft=self.nfft, noverlap=self.overlap,
                                              nperseg=self.nfft, detrend=False, window='hamming')
        return 20 * np.log(Sxx + 0.00000001)

    def getFeatureSet(self, freq1=None):
        # get spec data
        if freq1 is None:
            freq1 = self.freq1
        spec = self.generateSpecgram()
        freqs = np.fft.fftfreq(self.nfft)
        fs_effective = self.sampleRate / (self.nfft - self.overlap)
        tbin = int(round(self.nfft * freq1 / self.sampleRate))
        buff_out = 1

        # get raw freq range around generated frequency
        freq_range_features = np.vstack((spec[tbin - self.brange:tbin - buff_out, :],
                                         spec[tbin + buff_out:tbin + self.brange, :]))

        num_samples = freq_range_features.shape[1]

        # create vector with IR sensor data as sequence
        ir_features = np.zeros((2, num_samples))
        if len(self.sgestures) > 0:
            for x in self.sgestures:
                if x[1] == 0 and x[2] == 0:
                    x[1], x[2] = -10.0, -10.0
                idx_effective = int(
                    round(x[0] / self.sampleRate * fs_effective))
                if idx_effective >= num_samples:
                    if (idx_effective - num_samples > fs_effective / 4):
                        print('gesture happens %.2fs after ending, truncating' % (
                            (idx_effective - num_samples) / fs_effective))
                    idx_effective = num_samples - 1
                ir_features[0, idx_effective] = x[1]
                ir_features[1, idx_effective] = x[2]
                # ir_features[2, idx_effective] = self.dirs[0]
                # ir_features[3, idx_effective] = self.dirs[1]
                # ir_features[4, idx_effective] = self.dirs[2]
                # ir_features[5, idx_effective] = self.dirs[3]

        # Remove unwanted audio frequencies
        max_size = int(fs_effective * self.max_seconds)
        if freq_range_features.shape[1] > max_size:
            elim = list(range(max_size, freq_range_features.shape[1]))
            freq_features = np.delete(freq_range_features, elim, axis=1)
            ir_features = np.delete(ir_features, elim, axis=1)
        elif freq_range_features.shape[1] < max_size:
            pad_width = max_size - freq_range_features.shape[1]
            pad_mat = np.zeros((freq_range_features.shape[0], pad_width))
            freq_features = np.hstack((freq_range_features, pad_mat))

            pad_mat = np.zeros((ir_features.shape[0], pad_width))
            ir_features = np.hstack((ir_features, pad_mat))
        else:
            freq_features = freq_range_features

        freq_range_features = freq_features
        num_samples = freq_range_features.shape[1]

        return (freq_range_features, ir_features)


class GestureData():
    def __init__(self, gest_set=1):
        if gest_set == 1:
            self.shortNames = ["flickr", "flicku", "flickd", "flickl",
                               "panl", "panr", "panu", "pand",
                               "dtap", "tap", "dclick", "click",
                               "slicel", "slicer",
                               "zooma", "zoomi",
                               "whip", "snap", "magicw",
                               "circle", "erase"]
        elif gest_set == 2:
            self.shortNames = ["flickr", "flicku", "flickd", "flickl",
                               "dtap", "dclick", "magicw", "circle", "erase"]
        elif gest_set == 3:
            self.shortNames = ["flickr", "flicku", "flickd", "flickl",
                               "panr", "panu", "pand", "panl", "erase"]
        elif gest_set == 4:
            self.shortNames = ["zoomi", "zooma", "magicw",
                               "panr", "panu", "pand", "panl", "erase"]
        elif gest_set == 5:
            self.shortNames = ["magicw", "slicel", "slicer", "whip"]

    # Check if the gesture name exists
    def map_name(self, file):
        for name in self.shortNames:
            if name in file:
                return name
        return None

    def is_valid_withzero(self, name, ir_data, num_samples):
        possible_zeros = ["dtap", "dclick", "tap", "click",
                          "zooma", "zoomi", "whip", "snap", "magicw"]
        if np.sum((ir_data)) <= -5 * num_samples and name not in possible_zeros:
            return False
        return True

    def get_user_data(self, user, verbose, **args):
        user_path = './data/User_%d' % (user)
        files = [join(user_path, f)
                 for f in listdir(user_path) if isfile(join(user_path, f))]
        gests = [Gesture(filename=f, **args) for f in files]
        features = []
        for gest in gests:
            if len(gest.rawData) < 1000:
                continue
            # extract features
            freq_data, ir_data = gest.getFeatureSet(freq1=12000)
            name = self.map_name(gest.file)

            if name is not None and self.is_valid_withzero(name, ir_data, gest.numDirSamples):
                # save features with metadata
                features.append({'features': np.vstack((freq_data, ir_data)),
                                 'file': gest.file,
                                 'name': name,
                                 'user': user
                                 })
            elif verbose>0:
                print("Skipping files")
                # shutil.move(gest.file, user_path + '/skipped_files/')

        return features

    def keras_format(self, features):
        # Encoding the gestures
        y = [item['name'] for item in features]
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = y.reshape((-1, 1))

        # Creating a 4D tensor with size:  (# of samples, row, col, dimension).
        # This is specific for tensorflow
        nb_samples = len(features)
        nb_rows = features[0]['features'].shape[0]
        nb_col = features[0]['features'].shape[1]
        input_shape = (nb_col, nb_rows, 1)

        x = np.empty([nb_samples, nb_col, nb_rows, 1])
        for i, item in enumerate(features):
            x[i, :, :, 0] = item['features'].T

        # Extracting the users
        user = [item['user'] for item in features]
        user = np.array(user)

        print("\nInput shape:{0}".format(x.shape))
        print("\nLabels shape:{0}".format(y.shape))
        # print("Users shape:{0}".format(user.shape))

        return x, y, user, input_shape, le

    def compile_data(self, nfft, overlap, brange, max_seconds=3, keras_format=True, baseline_format=True,
                     plot_spectogram=True, verbose=0):
        features = []
        for user in [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15]:
            print("Collecting data for User: ", user)
            features += self.get_user_data(user, verbose, load=True, nfft=nfft,
                                           overlap=overlap, brange=brange, max_seconds=max_seconds)
        features = sorted(features, key=lambda k: k['name'])

        if plot_spectogram:
            return features
        elif keras_format:
            x_keras, y_keras, user, input_shape, lab_enc = self.keras_format(features)

            # normalize spec grams
            x_keras[:, :, :-2, :] = (x_keras[:, :, :-2, :] - np.mean(x_keras[:, :, :-2, :])) / np.std(
                x_keras[:, :, :-2, :])

            # normalize the ir values
            x_keras[:, :, -2, :] = (x_keras[:, :, -2, :] - np.mean(x_keras[:, :, -2, :])) / np.std(x_keras[:, :, -2, :])
            x_keras[:, :, -1, :] = (x_keras[:, :, -1, :] - np.mean(x_keras[:, :, -1, :])) / np.std(x_keras[:, :, -1, :])

            print("Number of classes: ", len(lab_enc.classes_))
            return x_keras, y_keras, user, input_shape, lab_enc
        elif baseline_format:
            x_baseline, y_baseline, user, lab_enc = self.baseline_format(features)
            return x_baseline, y_baseline, user, lab_enc
        else:
            raise ValueError('Could not find data format option to return')

    def baseline_format(self, features):
        # Encoding the gestures
        y = [item['name'] for item in features]
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = y.reshape(-1)

        nb_samples = len(features)
        nb_rows = features[0]['features'].shape[0] - 2
        nb_col = features[0]['features'].shape[1]

        x = np.empty([nb_samples, nb_col * nb_rows + 2])
        for i, item in enumerate(features):
            feat = item['features'].T
            doppler_feature = feat[:, :-2].reshape(-1)  # Flatten out doppler signature values
            ir1 = feat[:, -1]
            # Compute average of non-zero elements
            # ir1 - angle measurement
            # ir2 - velocity measurement
            if ir1.sum() != 0:
                ir1_feature = np.true_divide(ir1.sum(), (ir1 != 0).sum())
            else:
                ir1_feature = 0

            ir2 = feat[:, -2]
            if ir2.sum() != 0:
                ir2_feature = np.true_divide(ir2.sum(), (ir2 != 0).sum())
            else:
                ir2_feature = 0

            x[i, :] = np.hstack([doppler_feature, ir1_feature, ir2_feature])

        # Extracting the users
        user = [item['user'] for item in features]
        user = np.array(user)

        print("\nInput shape:{0}".format(x.shape))
        print("\nLabels shape:{0}".format(y.shape))

        return x, y, user, le
