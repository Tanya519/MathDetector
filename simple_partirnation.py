import os
import cv2
import numpy as np

### CHECK IT BEFORE RUNNING ###
dataset_way = 'dataset/'
image_name = 'test.jpg'
letter_size = 32
###############################


def in_letter(mas, coord):

    ### mas: array of coordinates of big letters
    ### coord: (x,y,w,h) coordinates of letter, which can intersect
    for i in mas:
        x = min(coord[0], i[0])
        y = min(coord[1], i[1])
        xx = max(coord[0] + coord[2], i[0] + i[2])
        yy = max(coord[1] + coord[3], i[1] + i[3])
        sq1 = i[2] * i[3]             ### square of big rectangle
        sq2 = (xx - x) * (yy - y)                       ### square of min rectangle, which contain both of rectangles
        if (sq1/sq2 >= 0.65):
            return(1)

    return(0)

def detector_on_page(page_name):
    img = cv2.imread(dataset_way + page_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output = img.copy()
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
            if not (in_letter(letters_coords, [x, y, w, h])):
                letter_count_in_page += 1

                letters_coords.append([x, y, w, h])

                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

                letter_crop = gray[y:y + h, x:x + w]

                size_max = max(w, h)
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                if w > h:
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop

                letters.append((x, w, cv2.resize(letter_square, (letter_size, letter_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    return letters, letter_count_in_page


letters, k = detector_on_page(image_name)
print('there are ', k,' letters')
for i in range(k):
    cv2.imshow(str(i), letters[i][2])
cv2.waitKey(0)