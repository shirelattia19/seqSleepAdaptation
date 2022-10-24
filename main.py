# Adaptation for PPG from Shirel Attia 20.10.22
import os
import math
import glob
import numpy as np
import scipy
import pandas as pd
from scipy.io import loadmat, savemat
from matplotlib import mlab
import matplotlib.pyplot as plt

from default_params import default_params
from preprocessing import preprocess_ppg


def nextpow2(x):
    return 0 if x == 0 else math.ceil(math.log2(x))


def plot_spec(Xk):
    plt.figure(figsize=(20, 20))
    plt.imshow(Xk)
    plt.savefig("myimage.png")
    plt.show()


def prepare_seqsleepnet_data(raw_data_path, signal_type, fs, win_size, overlap, nfft):
    extension = '.mat' if signal_type == 'eeg' else '.npy'
    for file in glob.glob(raw_data_path + '/*' + extension):
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
            y = np.zeros((label.shape[0], 6))
            for i, l in enumerate(label):
                y[i][int(l[0])] = 1
        else:
            print('Argument error: signal_path')
            exit(1)
        N = signal_epochs.shape[0]
        X = np.zeros((N, 29, int(nfft / 2 + 1)))
        for k in range(signal_epochs.shape[0]):
            if k % 100 == 0:
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
        savemat(os.path.join(raw_data_path, '..', f'mat_{signal_type}', os.path.basename(file).split('.')[0]+'.mat'),
                {'X': X, 'label': label, 'y': y})
        d = loadmat(os.path.join(raw_data_path, '..', f'mat_{signal_type}', os.path.basename(file).split('.')[0]+'.mat'))
        check = d['X']


def main():
    fs = 100
    win_size = 2
    overlap = 1
    nfft = 2 ** nextpow2(win_size * fs)
    #prepare_seqsleepnet_data(os.path.join('test', 'raw_data_eeg'), 'eeg', fs, win_size, overlap, nfft)
    prepare_seqsleepnet_data(os.path.join('test', 'raw_data_ppg'), 'ppg', fs, win_size, overlap, nfft)


if __name__ == '__main__':
    main()
