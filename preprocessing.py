import os
import traceback
import numpy as np
import pandas as pd
from scipy.signal import butter, resample, sosfiltfilt, cheby2

from dataset_tools import truncate_1D, sleep_extract_30s_epochs


def preprocess_ppg(path_signal, xml_path, params):
    ppg_signal = np.load(path_signal)
    mesa_signal = MESA_Signal(params)
    return mesa_signal.load_features(ppg_signal, xml_path)


class MESA_Signal():
    # PPG signal
    def __init__(self, params):
        # Setup the filtering parameters
        self.params = params

        fs = 256  # PPG source

        if params['filter_type'] == 'butter_bandpass':
            self.sos = butter(params['filter_order'],
                              [params['filter_lowcut'] / (fs / 2), params['filter_highcut'] / (fs / 2)], 'bandpass',
                              output='sos')
        elif params['filter_type'] == 'cheby_lowpass':
            self.sos = cheby2(params['filter_order'], params['ripple'], params['filter_highcut'] / (fs / 2), 'lowpass',
                              output='sos')
        # Setup the resampling parameters

        resample_fs = params['resample_fs']
        if not fs % resample_fs == 0:
            self.slow_resample = True
        else:
            self.slow_resample = False
        self.resample_factor = fs / resample_fs

    def _filter_signal(self, signal):
        return sosfiltfilt(self.sos, signal, axis=0)

    def _resample_with_filter(self, signal):
        signal = self._filter_signal(signal)
        if self.slow_resample:
            signal = resample(signal, int(len(signal) / self.resample_factor)).astype(np.float32)
        else:
            x_resample = np.arange(0, signal.shape[0], self.resample_factor, dtype='int32')
            signal = signal[x_resample]
        return signal

    def _rms_normalize(self, signal):
        signal = signal - np.mean(signal)
        max_level = np.quantile(np.abs(signal), 0.99)
        signal_99 = signal[np.abs(signal) > max_level]
        rms = np.sqrt(np.sum(np.square(signal_99)) / len(signal_99))
        signal_rms = signal / (rms * 10)
        return signal_rms

    def load_features(self, signal, xml_path):
        try:
            # Load, filter and resample the signal
            signal = self._resample_with_filter(signal)
            signal = self._rms_normalize(signal)

            # if it is more than 10 hours it removes the end of the signal
            epochs = int(signal.shape[0] / self.params['samples_per_epoch'])
            epochs = min(epochs, self.params['max_epochs'])
            samples = int(self.params['samples_per_epoch'] * epochs)
            signal = truncate_1D(signal, samples)
            X = signal.reshape(epochs, self.params['samples_per_epoch']).astype('float16')
            assert X.shape[0] <= self.params['max_epochs']

            Y = sleep_extract_30s_epochs(xml_path)
            Y = truncate_1D(Y, epochs)

            n_features = X.shape[1]
            x = np.zeros((1, epochs, n_features), dtype='float32')

            try:
                x[0, 0:epochs, :] = X[0:epochs, :]
            except:
                print("Failed to preprocess the patient")
                exit(1)

            x = (x - self.params['train_X_mean']) / self.params['train_X_std']

            # Apply updates to the dictionary
            return x[0], Y

        except:
            print('Failed to process')
            print(traceback.print_exc())
