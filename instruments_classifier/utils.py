import os
import shutil

import librosa
import numpy as np


def clear_folder(folder: str):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def copy_files_content(src: str, dest: str):
    for filename in os.listdir(src):
        shutil.copy(os.path.join(src, filename), dest)


def copy_folders_content(src: str, dest: str):
    for foldername in os.listdir(src):
        shutil.copytree(os.path.join(src, foldername),
                        os.path.join(dest, foldername))


def get_audio_features(path: str, duration=None, offset=None):
    y, sr = librosa.load(path, duration=duration, offset=offset)

    features = {
        # "spectrogram": librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1),
        # "mfccs": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),
        # "zcr": librosa.feature.zero_crossing_rate(y).mean(),
        # "chroma": librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
        # "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        # "rmse": librosa.feature.rms(y=y).mean()
    }

    feature_vector = np.concatenate(
        [features[key].flatten() for key in features])
    # print(feature_vector.shape)
    return feature_vector


def get_mfccs_feature(path: str, duration=None, offset=None):
    y, sr = librosa.load(path, duration=duration, offset=offset)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs
