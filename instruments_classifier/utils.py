import os
import shutil

import librosa
import numpy as np
from params import train_audio_duration


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


def get_audio_features(path: str, offset=None):
    y, sr = librosa.load(path, duration=train_audio_duration, offset=offset)

    features = {
        # "spectrogram": librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1),
        "mfccs": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),
        # "zcr": librosa.feature.zero_crossing_rate(y).mean(),
        # "chroma": librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
        # "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        # "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        # "rmse": librosa.feature.rms(y=y).mean()
    }

    feature_vector = np.concatenate(
        [features[key].flatten() for key in features])
    return feature_vector
