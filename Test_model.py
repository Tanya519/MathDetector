from keras.models import load_model
from PIL import Image
import numpy as np
import os

### CHECK IT BEFORE RUNNING ###

dataset_way = 'partirnated_dataset/'       ### way of dataset folder
model_name = 'good_model.hdf5'
###############################


def get_names():
    names = os.listdir('./' + dataset_way)
    names.sort()
    return names


def Test_model(model, w, h):
    X = load_image(w, h)
    y = model.predict_classes(X)
    print(y)


def load_image(w, h):
    names = get_names()
    pict_count = len(names)
    pixels = np.zeros(pict_count * w * h * 3).reshape(pict_count, w, h, 3)

    for pict in range(len(names)):
        name = dataset_way + names[pict]
        img_start = Image.open(name)
        img = Image.new("RGBA", img_start.size)
        img.paste(img_start)
        img = img.resize((w, h))
        pix = img.load()
        for line in range(h):
            for column in range(w):
                pixels[pict, line, column, 0] = np.array(pix[line, column][0])
                pixels[pict, line, column, 1] = np.array(pix[line, column][1])
                pixels[pict, line, column, 2] = np.array(pix[line, column][2])

    pixels = pixels.astype('float32')
    pixels = pixels / 255.0
    return pixels


model = load_model(model_name)
Test_model(model, 32, 32)