import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
import joblib

from .utils import get_audio_features
from params import data_dirs, max_audios, train_audio_duration


def run():
    X = []
    y_labels = []
    y = []
    for i, instrument_dir in zip(range(len(data_dirs)), data_dirs):
        y_labels.append(instrument_dir)
        print(f'Reading {instrument_dir} audios...')
        audio_paths = os.listdir(f'data/{instrument_dir}')

        readed_audios = 0
        for audio_path in audio_paths:
            path = f'data/{instrument_dir}/{audio_path}'
            duration = librosa.get_duration(path=path)
            offset = 0
            while offset <= duration - train_audio_duration:
                data = get_audio_features(
                    path, duration=train_audio_duration, offset=offset)
                X.append(data)
                y.append(i)
                offset += train_audio_duration
                readed_audios += 1

            if max_audios != -1 and readed_audios >= max_audios:
                break

    print('Training model...')

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    joblib.dump(clf, 'svm_model.joblib')
    joblib.dump(y_labels, 'labels.joblib')
