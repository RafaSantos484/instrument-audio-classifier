import os
import joblib
import numpy as np
from sklearn.svm import SVC

from .utils import get_audio_features


def run():
    clf: SVC = joblib.load('svm_model.joblib')
    inputs = []
    filenames = os.listdir('test')
    input_size = clf.n_features_in_
    labels = joblib.load('labels.joblib')
    for filename in filenames:
        data = get_audio_features(f'test/{filename}')
        inputs.append(data)

    inputs = np.array(inputs)
    predictions = clf.predict(inputs)

    for pred, filename in zip(predictions, filenames):
        print(f'{filename}: {labels[pred]}')
