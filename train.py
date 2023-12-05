import numpy as np
import librosa
import os
import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import LambdaCallback
from sklearn.metrics import classification_report, confusion_matrix

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

def print_accuracy(epoch, logs):
    global accuracy
    if epoch == 99:
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def print_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

accuracy = ''
krs_clb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
accuracy_cb = LambdaCallback(on_epoch_end=print_accuracy)

X_train, X_test, y_train, y_test, model = full_data()

print(model.summary())

history = model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1, validation_split=0.1, callbacks=[krs_clb, accuracy_cb])

print(f'Accuracy: {round(accuracy * 100, 2)} %')

plot_accuracy(history)
y_pred = np.argmax(model.predict(X_test), axis=1)
class_labels = np.argmax(y_test, axis=1)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
plot_confusion_matrix(class_labels, y_pred, class_names)
print_classification_report(class_labels, y_pred, class_names)

model.save('model.h5')
