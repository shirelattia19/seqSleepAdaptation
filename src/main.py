# Adaptation for PPG from Shirel Attia 20.10.22
import os
import math
import glob
import numpy as np
import h5py as h5
from scipy.io import loadmat, savemat
from matplotlib import mlab
import matplotlib.pyplot as plt

from src.ppg_preprocessing.default_params import default_params
from src.ppg_preprocessing.preprocessing import preprocess_ppg


def nextpow2(x):
    return 0 if x == 0 else math.ceil(math.log2(x))


def plot_spec(Xk):
    plt.figure(figsize=(20, 20))
    plt.imshow(Xk)
    plt.savefig("myimage.png")
    plt.show()


def prepare_seqsleepnet_data(raw_data_path, signal_type, fs, win_size, overlap, nfft):
    # See file prepare_seqsleepnet_data.m from huy's code
    extension = '.mat' if signal_type == 'eeg' else '.npy'
    for file in glob.glob(raw_data_path + '/*' + extension):
        print(f'-------------------------- Computing file {file} --------------------------')
        # label and one-hot encoding
        if signal_type == 'eeg':
            d = loadmat(file)
            data = d['data']
            y = np.double(d['labels'])
            label = np.where(y == 1)[1].reshape((y.shape[0], 1))
            signal_epochs = np.squeeze(data[:, :, 0])
        elif signal_type == 'ppg':
            signal_epochs, labels = preprocess_ppg(file, file.replace('PPG.npy', 'nsrr.xml'), default_params['dataset'])
            label = labels.reshape((labels.shape[0], 1))
            y = np.zeros((label.shape[0], 5))
            for i, l in enumerate(label):
                y[i][int(l[0])] = 1
        else:
            print('Argument error: signal_path')
            exit(1)
        N = signal_epochs.shape[0]
        X = np.zeros((N, 29, int(nfft / 2 + 1)))
        for k in range(signal_epochs.shape[0]):
            if k % 300 == 0:
                print(k, '/', signal_epochs.shape[0])
            Xk, _, _ = mlab.specgram(x=signal_epochs[k, :], pad_to=nfft, NFFT=fs * win_size, Fs=fs,
                                     window=np.hamming(fs * win_size), noverlap=overlap * fs, mode='complex')

            # _, _, Xk2 = scipy.signal.spectrogram(eeg_epochs[k, :], fs=fs, nperseg=fs * win_size, noverlap=overlap * fs,
            #                                    nfft=nfft, mode='complex')
            Xk = 20 * np.log10(np.abs(Xk))
            gfg = np.matrix(Xk)
            Xk = gfg.getH()
            # plot_spec(Xk)
            X[k, :, :] = Xk
        nans = np.unique(np.isnan(X))
        infs = np.unique(np.isinf(X))
        if (len(nans) > 1) or (len(infs) > 1) or infs[0] or nans[0]:
            print('Error NAN/INF values, not saving the data. ')
        else:
            saved_file = h5.File(
                os.path.join(raw_data_path, '..', f'mat_{signal_type}', os.path.basename(file).split('.')[0] + '.mat'), 'w')
            saved_file.create_dataset('X', data=X)
            saved_file.create_dataset('y', data=y)
            saved_file.create_dataset('label', data=label)
            saved_file.close()

            d = h5.File(os.path.join(raw_data_path, '..', f'mat_{signal_type}', os.path.basename(file).split('.')[0]+'.mat'), 'r')
            d.keys()
            check = d.get('X')
            d.close()


def main1():
    fs = 100
    win_size = 2
    overlap = 1
    nfft = 2 ** nextpow2(win_size * fs)
    #prepare_seqsleepnet_data(os.path.join('test', 'raw_data_eeg'), 'eeg', fs, win_size, overlap, nfft)
    prepare_seqsleepnet_data(os.path.join('test', 'raw_data_ppg'), 'ppg', fs, win_size, overlap, nfft)

def main():
    #data = h5.File(os.path.join('test', 'mat_ppg', 'mesa-sleep-6704-PPG.mat'), 'r')
    data = h5.File(os.path.join('test', 'mat_eeg', 'SS1_01-01-0001_cnn_filterbank_eeg.mat'), 'r')

    data.keys()
    X = np.array(data['X'])
    X = np.transpose(X, (2, 1, 0))  # rearrange dimension
    y = np.array(data['y'])
    y = np.transpose(y, (1, 0))  # rearrange dimension
    label = np.array(data['label'])
    label = np.transpose(label, (1, 0))  # rearrange dimension
    label = np.squeeze(label)


if __name__ == '__main__':
    main()
