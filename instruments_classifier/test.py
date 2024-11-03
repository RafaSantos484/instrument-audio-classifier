import os
import joblib
import numpy as np
import librosa
import tensorflow as tf

from .utils import get_audio_features, get_mfccs_feature
from params import train_audio_duration, test_model


def get_classifier():
    if test_model == 'svm':
        return joblib.load('svm_model.joblib')
    else:
        return tf.keras.models.load_model('cnn_model.h5')


def run():
    clf = get_classifier()
    inputs = []
    filenames = []
    preds = {}
    labels = joblib.load('labels.joblib')
    for filename in os.listdir('test'):
        preds[filename] = np.zeros(len(labels))
        duration = librosa.get_duration(path=f'test/{filename}')
        offset = 0
        while offset <= duration - train_audio_duration and offset <= 60:
            if test_model == 'svm':
                feature = get_audio_features(
                    f'test/{filename}', duration=train_audio_duration, offset=offset)
            else:
                feature = get_mfccs_feature(
                    f'test/{filename}', duration=train_audio_duration, offset=offset)

            inputs.append(feature)
            filenames.append(filename)
            offset += train_audio_duration

    inputs = np.array(inputs)
    predictions = clf.predict(inputs)

    for pred, filename in zip(predictions, filenames):
        if test_model == 'cnn':
            pred = np.argmax(pred)
        preds[filename][pred] += 1

    for filename, pred_array in preds.items():
        pred_class = np.argmax(pred_array)
        percentage = pred_array[pred_class] / np.sum(pred_array) * 100
        print(f'{filename}: {labels[pred_class]} ({percentage:.2f}%)')
