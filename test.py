import keras
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def Model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)), 
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

def convert_wav(file_path, max_pad_len=20):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    m = librosa.feature.mfcc(y=wave, sr=16000)
    pad_width = max_pad_len - m.shape[1]
    m = np.pad(m, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return m

def dataset_data_extract():
    labels = []
    mfccs = []
    for f in os.listdir('./dataset'):
        if f.endswith('.wav'):
            mfccs.append(convert_wav('./dataset/' + f))
            label = f.split('_')[0]
            labels.append(label)
    return np.asarray(mfccs), to_categorical(labels)

def full_data():
    mfccs, labels = dataset_data_extract()
    dim_1 = mfccs.shape[1]
    dim_2 = mfccs.shape[2]
    channels = 1
    num_classes = labels.shape[1]
    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels
    input_shape = (dim_1, dim_2, channels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    model = Model(input_shape, num_classes)
    return X_train, X_test, y_train, y_test, model

def test_data_for_predictions(X, y):
    trained_model = keras.models.load_model('model.h5')
    predictions = trained_model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    print(classification_report(y, to_categorical(predicted_classes)))


if __name__ == '__main__':
    _, X_test, _, y_test, _ = full_data()
    test_data_for_predictions(X_test, y_test)
