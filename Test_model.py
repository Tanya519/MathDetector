from keras.models import load_model
from PIL import Image
import numpy as np
import os

dataset_way = 'test_dataset/'

def get_names():
    names = os.listdir('./' + dataset_way)
    names.sort()
    return names


def Test_model(model, w, h):
    # X = load_image(w, h)
    X = load_image(w, h)
    y = model.predict_classes(X)
    print(y)
    print('hoba')


def load_image(w, h):
    names = get_names()
    pict_count = len(names)
    pixels = np.zeros(pict_count * w * h * 3).reshape(pict_count, w, h, 3)

    for pict in range(len(names)):
        name = dataset_way + names[pict]
        img = Image.open(name)
        img = img.resize((w, h))
        pix = img.load()

        for line in range(h):
            for column in range(w):
                pixels[pict, line, column, 0] = np.array(pix[line, column][0])
                pixels[pict, line, column, 1] = np.array(pix[line, column][1])
                pixels[pict, line, column, 2] = np.array(pix[line, column][2])

    return pixels


model = load_model('number_recognition_0.1.hdf5')
Test_model(model, 32, 32)
