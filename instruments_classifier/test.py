import os
import joblib
import numpy as np
from sklearn.svm import SVC
import librosa

from .utils import get_audio_features
from params import train_audio_duration


def run():
    clf: SVC = joblib.load('svm_model.joblib')
    inputs = []
    filenames = []
    preds = {}
    labels = joblib.load('labels.joblib')
    for filename in os.listdir('test'):
        preds[filename] = np.zeros(len(labels))
        duration = librosa.get_duration(path=f'test/{filename}')
        offset = 0
        while offset <= duration - train_audio_duration and offset <= 60:
            data = get_audio_features(
                f'test/{filename}', duration=train_audio_duration, offset=offset)
            inputs.append(data)
            filenames.append(filename)
            offset += train_audio_duration

    inputs = np.array(inputs)
    predictions = clf.predict(inputs)

    for pred, filename in zip(predictions, filenames):
        preds[filename][pred] += 1

    for filename, pred_array in preds.items():
        pred_class = np.argmax(pred_array)
        percentage = pred_array[pred_class] / np.sum(pred_array) * 100
        print(f'{filename}: {labels[pred_class]} ({percentage:.2f}%)')
