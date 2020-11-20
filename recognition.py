import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import os
import sys

def get_names():
    names = os.listdir('./dataset/')
    names.sort()
    return names


def make_Y(X):
    k = each_letter_count
    y_t = np.zeros(int(len(X) * percentage_in_test))
    y = np.zeros(len(X) - int(len(X) * percentage_in_test))

    p1 = 0
    p2 = 0
    target = 0
    for i in range(len(X)):
        if i % k < k * percentage_in_test:
            y_t[p1] = target
            p1 += 1
        else:
            y[p2] = target
            p2 += 1
        if (i + 1) % k == 0:
            target += 1
    return y, y_t


def print_percentage(p):
    perc = int(p*100)
    sys.stdout.write("\r" + str(perc) + "% done ")
    sys.stdout.flush()


def make_dataset(w, h):
    names = get_names()
    test_pict_count = int(len(names) * percentage_in_test)
    pict_count = int(len(names) - test_pict_count)
    Pixels_info = np.zeros(pict_count * w * h * 3).reshape(pict_count, w, h, 3)
    test_Pixels_info = np.zeros(test_pict_count * w * h * 3).reshape(test_pict_count, w, h, 3)
    p1 = -1
    p2 = -1

    print('preparing datasets...')
    for pict in range(len(names)):
        name = 'dataset/' + names[pict]
        img = Image.open(name)
        img = img.resize((w, h))
        pix = img.load()
        print_percentage(pict/len(names))
        if pict % each_letter_count < each_letter_count * percentage_in_test:
            p1 += 1
            for line in range(h):
                for column in range(w):
                    test_Pixels_info[p1, line, column, 0] = np.array(pix[line, column][0])
                    test_Pixels_info[p1, line, column, 1] = np.array(pix[line, column][1])
                    test_Pixels_info[p1, line, column, 2] = np.array(pix[line, column][2])
        else:
            p2 += 1
            for line in range(h):
                for column in range(w):
                    Pixels_info[p2, line, column, 0] = np.array(pix[line, column][0])
                    Pixels_info[p2, line, column, 1] = np.array(pix[line, column][1])
                    Pixels_info[p2, line, column, 2] = np.array(pix[line, column][2])
    y_train, y_test = make_Y(names)

    print("datasets created")
    return Pixels_info, y_train, test_Pixels_info, y_test


def letter_recognition(X_train, y_train, X_test, y_test):
    ### Prepare dataset

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    class_num = y_test.shape[1]

    ### Prepare model
    print('preparing model...')
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(512, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    epochs = 10
    print('model created')

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[
            'accuracy',
            'AUC',
        ]
    )

    print(model.summary())

    np.random.seed(seed)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=20)

    # Final evaluation of the model

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    ### save model ###
    model.save('number_recognition.hdf5')
    ##################

seed = 21
each_letter_count = 3250
percentage_in_test = 0.1

X_train, y_train, X_test, y_test = make_dataset(32, 32)
letter_recognition(X_train, y_train, X_test, y_test)