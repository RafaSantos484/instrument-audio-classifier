import os
import numpy as np
from sklearn.model_selection import train_test_split
import librosa
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from .utils import get_cnn_feature
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
                feature = get_cnn_feature(
                    path, duration=train_audio_duration, offset=offset)
                X.append(feature)
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

    print(X.shape[1:] + (1,))
    model = Sequential([
        Input(shape=X.shape[1:] + (1,)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(data_dirs), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10)
    _, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')

    model.save('cnn_model.keras')
    joblib.dump(y_labels, 'labels.joblib')
