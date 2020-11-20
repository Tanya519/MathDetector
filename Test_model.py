from keras.models import load_model
from PIL import Image
import numpy as np
import os

def get_names():
    names = os.listdir('./test_dataset/')
    names.sort()
    return names

def Test_model(model, w, h):
    X = load_image(w, h)
    y = np.argmax(model.predict(X), axis=-1)
    print(y)

def load_image(w,h):
    names = get_names()
    pict_count = len(names)
    pixels = np.zeros(pict_count * w * h * 3).reshape(pict_count, w, h, 3)

    for pict in range(len(names)):
        name = 'test_dataset/' + names[pict]
        img = Image.open(name)
        img = img.resize((w, h))
        pix = img.load()

        for line in range(h):
            for column in range(w):
                pixels[pict, line, column, 0] = np.array(pix[line, column][0])
                pixels[pict, line, column, 1] = np.array(pix[line, column][1])
                pixels[pict, line, column, 2] = np.array(pix[line, column][2])

    return pixels

model = load_model('number_recognition.hdf5')
Test_model(model, 32, 32)