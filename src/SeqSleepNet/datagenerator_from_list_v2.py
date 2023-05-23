# sequence sampling and minibatch generation for network training
import numpy as np
import os
from scipy.io import loadmat
import h5py

class DataGenerator:
    def __init__(self, filelist, data_shape=np.array([29, 129]), seq_len = 32, shuffle=False, test_mode=False):

        # Init params
        self.shuffle = shuffle
        self.filelist = filelist
        self.data_shape = data_shape
        self.pointer = 0
        self.X = np.array([])
        self.y = np.array([])
        self.label = np.array([])
        self.boundary_index = np.array([])
        self.test_mode = test_mode

        self.seq_len = seq_len
        self.Ncat = 5
        # read from mat file

        self.read_mat_filelist(self.filelist)
        
        if self.shuffle:
            self.shuffle_data()

    def read_mat_filelist(self,filelist):
        """
        Scan the file list and read them one-by-one
        """
        files = []
        self.data_size = 0
        with open(filelist) as f:
            lines = f.readlines()
            for l in lines:
                #print(l)
                items = l.split()
                files.append(items[0])
                self.data_size += int(items[1])
                print(self.data_size)
        self.X = np.ndarray([self.data_size, self.data_shape[0], self.data_shape[1]])
        self.y = np.ndarray([self.data_size, self.Ncat])
        self.label = np.ndarray([self.data_size])
        count = 0
        for i in range(len(files)):
            X, y, label = self.read_mat_file(files[i].strip())
            print(f'len(X) = {len(X)}')
            print(f'shape(X) = {X.shape}')
            print(f'count = {count}')
            self.X[count : count + len(X)] = X
            self.y[count : count + len(X)] = y
            self.label[count : count + len(X)] = label
            # indexes at the end of each recording which do not constitute a full sequence will be excluded
            self.boundary_index = np.append(self.boundary_index, np.arange(count, count + self.seq_len - 1))
            count += len(X)
            print(count)

        print("Boundary indices")
        print(self.boundary_index)
        self.data_index = np.arange(len(self.X))
        print(len(self.data_index))
        # excluding those indexes in self.boundary_index here
        if self.test_mode == False:
            mask = np.in1d(self.data_index,self.boundary_index, invert=True)
            self.data_index = self.data_index[mask]
            print(len(self.data_index))
        print(self.X.shape, self.y.shape, self.label.shape)

    def read_mat_file(self,filename):
        """
        Read matfile HD5F file and parsing
        """
        # Load data from .mat files
        data = h5py.File(filename,'r')
        data.keys()
        X = np.array(data['X'])
        #X = np.transpose(X, (2, 1, 0))  # rearrange dimension
        y = np.array(data['y'])
        #y = np.transpose(y, (1, 0))  # rearrange dimension
        label = np.array(data['label'])
        #label = np.transpose(label, (1, 0))  # rearrange dimension
        label = np.squeeze(label)

        return X, y, label

        
    def shuffle_data(self):
        """
        Random shuffle the data points indexes
        """
        idx = np.random.permutation(len(self.data_index))
        self.data_index = self.data_index[idx]

                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        # shuffle data again
        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        This function generates a minibatch of data
        """
        data_index = self.data_index[self.pointer:self.pointer + batch_size]

        #update pointer
        self.pointer += batch_size

        # in case multiple channels are used, after stacking (e.g. eeg, eog, emg), data_shape now is 3D
        # in case single channel are used, the channel dimension is expanded, data_shape now is also 3D
        batch_x = np.ndarray([batch_size, self.seq_len, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
        batch_y = np.ndarray([batch_size, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([batch_size, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x[i, n]  = self.X[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        batch_x.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        return batch_x, batch_y, batch_label

    def rest_batch(self, batch_size):
        """
        This function generates the remaining data which do not constitute a full minibatch. It is necessary when evaluation
        """

        # actual length of this mini batch
        data_index = self.data_index[self.pointer : len(self.data_index)]
        actual_len = len(self.data_index) - self.pointer

        # update pointer
        self.pointer = len(self.data_index)

        # in case multiple channels are used, after stacking (e.g. eeg, eog, emg), data_shape now is 3D
        # in case single channel are used, the channel dimension is expanded, data_shape now is also 3D
        batch_x = np.ndarray([actual_len, self.seq_len, self.data_shape[0], self.data_shape[1], self.data_shape[2]])
        batch_y = np.ndarray([actual_len, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([actual_len, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x[i, n]  = self.X[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]
            # check condition to make sure all corrections
            #assert np.sum(batch_y[i]) > 0.0

        batch_x.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        return actual_len, batch_x, batch_y, batch_label
