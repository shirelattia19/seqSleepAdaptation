import numpy as np
from numpy.core._multiarray_umath import ndarray
import xml.etree.ElementTree as ET

def time_series_subsequences(ts, window, hop=1):
    assert len(ts.shape) == 1
    shape = (int(int(ts.size - window) / hop + 1), window)
    strides = ts.strides[0] * hop, ts.strides[0]
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

def subsequenced_padded(X, win_len, hop):
    assert X.shape[0] % hop == 0
    padding = int(win_len / 2 - hop / 2)
    if padding>0:
        X__ = np.zeros((X.shape[0] + 2 * padding))
        X__[padding:-padding] = X
    else:
        X__ = X
    X_ = time_series_subsequences(X__.flatten(), win_len, hop)
    return X_

def batch_time_series_subsequences(batch, window, hop):
    shape = time_series_subsequences(batch[0], window, hop).shape
    strided_batch = np.empty(shape=(batch.shape[0], shape[0], shape[1]), dtype=batch.dtype)
    for i in range(batch.shape[0]):
        strided_batch[i] = time_series_subsequences(batch[i], window, hop)
    return strided_batch

def truncate_1D(sig_in, max_len):
    assert len(sig_in.shape) == 1
    if sig_in.shape[0] > max_len:
        return sig_in[0:max_len]
    else:
        return sig_in

def sleep_extract_30s_epochs(path):
    try:
        root = ET.parse(path).getroot()
        events = [x for x in list(root.iter()) if x.tag == "ScoredEvent"]
        events_decomposed = list([list(event.iter()) for event in events])
        stage_events = [x for x in events_decomposed if x[1].text == "Stages|Stages"]
        starts = np.array([float(x[3].text) for x in stage_events]) / 30
        durations = np.array([float(x[4].text) for x in stage_events]) / 30
        stages = np.array([int(x[2].text[-1]) for x in stage_events])
        sleep_timeline: ndarray = np.zeros(int(starts[-1] + durations[-1]))
        for i in range(len(stages)):
            sleep_timeline[int(starts[i]): int(starts[i] + durations[i])] = stages[i]
        return sleep_timeline
    except:
        print(f"Could not extract sleep from: {path}")
        return 0
