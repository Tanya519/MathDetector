from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import pandas as pd
import numpy as np
import random

import os.path
import random
import string

FONTS_PATH = r"E:\Proga\Project\Fonts"
FONT_SUFFIX = '.ttf'
ttf_files = [name for name in os.listdir(FONTS_PATH) if name.endswith(FONT_SUFFIX)]


def salt_and_pepper(image, prob=0.05):
    if prob <= 0:
        return image

    arr = np.asarray(image)
    original_dtype = arr.dtype

    intensity_levels = 2 ** (arr[0, 0].nbytes * 8)

    min_intensity = 0
    max_intensity = intensity_levels - 1

    random_image_arr = np.random.choice(
        [min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape
    )

    salt_and_peppered_arr = arr.astype(np.float) * random_image_arr

    salt_and_peppered_arr = np.nan_to_num(
        salt_and_peppered_arr, max_intensity
    ).astype(original_dtype)

    return Image.fromarray(salt_and_peppered_arr)

class DatasetItem_degree:
    def __init__(self, value, degree, basename):
        self.filenames_ = []
        self.value_ = value
        self.degree_ = degree
        self.basename_ = basename

    def generate(self, target_dir):
        if not os.path.exists(target_dir):
            raise ValueError(f'target dir {target_dir} does not exist')

        index = 0

        # print(ttf_files)
        for font_file in ttf_files:
            img = Image.new('RGBA', (len(self.value_) * 40 + 200, 200), color="white")
            draw = ImageDraw.Draw(img)
            x_coord = random.randint(0, 30)
            y_coord = random.randint(10, 100)
            wsize = random.randint(80, 150)
            draw.text((x_coord, y_coord), self.value_, font=ImageFont.truetype(os.path.join(FONTS_PATH, font_file),
                                                                               size=wsize), fill="gray")
            if self.degree_:
                draw.text((x_coord + 15, y_coord - 10), self.degree_,
                          font=ImageFont.truetype(os.path.join(FONTS_PATH, font_file), size=int(wsize // 1.5)),
                          fill="gray")
            self.filenames_.append(f'{self.basename_}_{index:04d}.jpg')

            salt_and_pepper(img, 0.02).convert('RGB').save(os.path.join(target_dir, self.filenames_[-1]), quality=95)
            index += 1

        return self


# .transform((200, 200), Image.AFFINE,
#       [1, -0.5, 0, 0, 1, 0], Image.BICUBIC).convert('RGB').save("output.jpg", quality=95)

# print(type(button_img.convert('RGB')))

items = []


def generate_digits():
    print('generating digits')
    for digit in range(10):
        items.append(DatasetItem_degree(str(digit), None, f"digit_{digit}").generate("dataset"))
    print('done')


def generate_lowercase():
    print('generating lowercase')
    for char in string.ascii_lowercase:
        items.append(DatasetItem_degree(char, None, f"letter_{char}").generate("dataset"))
    print('done')


def generate_uppercase():
    print('generating uppercase')
    for char in string.ascii_uppercase:
        items.append(DatasetItem_degree(char, None, f"letter_cap_{char}").generate("dataset"))
    print('done')


def generate_signs():
    print('generating signs')
    named_signs = {'+': 'plus',
                   '-': 'minus',
                   '*': 'asterisc',
                   '=': 'equals',
                   }
    for sign in named_signs:
        items.append(DatasetItem_degree(sign, None, f"sign_{named_signs[sign]}").generate("dataset"))
    print('done')


def generate_expressions():
    print('generating expressions')
    expressions = [
        'x+y', '2+2=4+2+2+2+2+2', '5=2 mod 3', '5%3=2'
    ]
    for index, expr in enumerate(expressions):
        items.append(DatasetItem_degree(expr, None, f"expr_{index:05d}").generate("dataset"))
    print('done')


def generate_degrees():
    print('generating degrees')
    for digit in range(10):
        for degree in range(10):
            items.append(DatasetItem_degree(str(digit), str(degree), f"digitt_{digit}_{degree}").generate("dataset"))


def metas():
    meta_pairs = [(e.value_, fn) for e in items for fn in e.filenames_]
    metadata = pd.DataFrame({'value': [e for e, _ in meta_pairs],
                             'filename': [e for _, e in meta_pairs],
                             })
    metadata.to_csv('metadata.csv', index=False)


generate_expressions()
metas()