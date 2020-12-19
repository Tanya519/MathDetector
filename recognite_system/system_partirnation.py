import cv2
import numpy as np


### ФУНКЦИИ ###
#   dist(x,y,x,y)   возвращает расстоение между точками
#
#   detector_on_page(name)  получает имя картинки, возвращает ее разбиение на буквы и количество букв
#
#   make_better_image(img)  получает картинку, возвращает ее копию с утолщенными контурами
#
#   to_pict(letters)    получает массив букв, по ним строит строки в картинке, возвращает их
#
#   partirnate_file(image_name) получает путь к картинке, находит там знак {, возвращает строки, который идут перед ним,
#   и строки, которые идут после него (двумя массивами)
#
### CHECK IT BEFORE RUNNING ###
dataset_way = '4test/'
way_to_save_letters = 'test_data/'
image_name = 'system3.png'
letter_size = 80
letter_indent = 10
img = cv2.imread(dataset_way + image_name)
pict_width = img.shape[0]
pict_height = img.shape[1]
###############################

def dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def detector_on_page(page_name):            ### detect letters on page, returns array of latters pics and count of latters
    img = cv2.imread(page_name)
    width = img.shape[1]
    height = img.shape[0]

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    letter_count_in_page = 0
    letters = []
    letters_coords = []

    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts:

        ### criterion for detecting letters ###

        if cv2.contourArea(cnt) > 10:
            (x, y, w, h) = cv2.boundingRect(cnt)

            ### checking intersections ###
            letter_count_in_page += 1
            w = w + min(int(letter_indent / 2), x) + min(int(letter_indent / 2), width - x - w)
            x = x - min(int(letter_indent / 2), x)
            h = h + min(int(letter_indent / 2), y) + min(int(letter_indent / 2), height - y - h)
            y = y - min(int(letter_indent / 2), y)
            s = w * h
            letters_coords.append([x, y, w, h, s])

    letters_coords.sort(key=lambda x: x[4], reverse=True)


    # croping letters
    for i in letters_coords:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]

        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

        letter_crop = gray[y:y + h, x:x + w]

        size_max = max(w, h)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

        letter_square = letter_crop

        letters.append((x, w, y, h, cv2.resize(letter_square, (w, h), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters, letter_count_in_page


def make_better_image(img):
    image = np.ones(img.shape) * 255
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i,j] == 0:
                image[i, j] = 0
                image[i-1, j] = 0
                image[i, j-1] = 0
                image[i+1, j] = 0
                image[i, j+1] = 0
                image[i - 1, j - 1] = 0
                image[i + 1, j - 1] = 0
                image[i + 1, j + 1] = 0
                image[i - 1, j + 1] = 0

    return image


def to_pict(letters):
    imgs = []

    for i in range(len(letters)):
        x = letters[i][0]
        w = letters[i][1]
        y = letters[i][2]
        h = letters[i][3]
        img = letters[i][4]

        inserted = 0

        for pic_num in range(len(imgs)):
            xx = imgs[pic_num][0]
            ww = imgs[pic_num][1]
            yy = imgs[pic_num][2]
            hh = imgs[pic_num][3]
            image = imgs[pic_num][4]
            if (dist(0,y,0,yy) <= w) or (y >= yy and y <= yy + hh):
                image[y:y+h, x:x+w] = img
                if y <= yy:
                    imgs[pic_num][2] = y
                imgs[pic_num][1] = x + w - xx
                imgs[pic_num][3] = max(y + h, yy + hh) - imgs[pic_num][2]
                inserted = 1
                break

        if (len(imgs) == 0) or (not inserted):
            image = np.ones((pict_width, pict_height)) * 255
            image[y:y+h, x:x+w] = img
            imgs.append([x, w, y, h, image])


    print(len(imgs))
    images = []
    for pic_num in range(len(imgs)):
        xx = imgs[pic_num][0]
        ww = imgs[pic_num][1]
        yy = imgs[pic_num][2]
        hh = imgs[pic_num][3]
        image = imgs[pic_num][4]

        #image_crop = make_better_image(image[yy : yy + hh, xx : xx + ww])
        image_crop = image[yy : yy + hh, xx : xx + ww]
        images.append([yy, image_crop])

    images.sort(key= lambda x: x[0], reverse=False)
    return images


def partirnate_file(image_name):
    letters, k = detector_on_page(image_name)
    max_height = 0              #detecting {
    ind_of_max = 0
    print('there are ', len(letters),' letters')
    for i in range(len(letters)):
        img = cv2.cvtColor(letters[i][4],cv2.COLOR_GRAY2RGB)
        #cv2.imshow(str(i), img)
        if img.shape[1] >= max_height:
            max_height = img.shape[1]
            ind_of_max = i


    ### we know, that equations startet with index ind_of_max + 1
    letters_after = letters[ind_of_max+1:]
    letters_before = letters[:ind_of_max]

    images_after = to_pict(letters_after)
    images_before = to_pict(letters_before)

    #for i in range(len(images_before)):
    #    cv2.imshow(str(i), images_before[i][1])
    #for i in range(len(images_after)):
    #    cv2.imshow(str(i+5), images_after[i][1])
    #cv2.waitKey(0)

    return images_before, images_after