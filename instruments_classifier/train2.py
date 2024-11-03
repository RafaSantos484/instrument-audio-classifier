import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from .utils import get_audio_features
from params import data_dirs, max_audios


def run():
    X = []
    y_labels = []
    y = []
    for i, instrument_dir in zip(range(len(data_dirs)), data_dirs):
        y_labels.append(instrument_dir)
        print(f'Reading {instrument_dir} audios...')
        audio_paths = os.listdir(f'data/{instrument_dir}')
        if max_audios != -1:
            audio_paths = audio_paths[:max_audios]
        for audio_path in audio_paths:
            X.append(get_audio_features(f'data/{instrument_dir}/{audio_path}'))
            y.append(i)

    print('Training model...')

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    joblib.dump(clf, 'svm_model.joblib')
    joblib.dump(y_labels, 'labels.joblib')
