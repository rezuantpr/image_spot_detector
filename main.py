# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2
from math import pi
import matplotlib.pyplot as plt


def detectSpots(image):
    cv2.destroyAllWindows()
    img = cv2.medianBlur(image, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=30, param2=15, minRadius=0, maxRadius=20)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for i, (x, y, r) in enumerate(circles):
            # draw the circle in the output image
            cv2.circle(cimg, (x, y), r, (0, 200, 0), 2)
            cv2.putText(cimg, str(r), (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(cimg, str(int(numberOfPixels(r))), (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # для наглядности рисуем индексы каждого пятна
            cv2.putText(cimg, str(i), (x, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return cimg, circles


def drawPlot(image, spot1, n1, spot2, n2):
    b_list = np.arange(0.1, 100, 0.1)
    f_list = []

    for b in b_list:
        f = f_func(b, image, spot1, n1, spot2, n2)
        f_list.append(f)

    plt.plot(b_list, f_list)

    plt.xlabel('b')
    plt.ylabel('f')
    plt.title('f(b)')

    # function to show the plot
    return plt


def process():
    image = cv2.imread("full.jpg", cv2.IMREAD_GRAYSCALE)

    (result_image, spots) = detectSpots(image)

    # алгоритм определяет заранее известные пятна под номерами 20 и 21
    # в будущем это исправлю, пока для теста так сойдет

    spot1 = spots[21]   # пятно с 1 г/м
    spot2 = spots[20]   # пятно с 2 г/м

    n1 = 1  # г/м ???
    n2 = 2  # г/м

    plt = drawPlot(image, spot1, n1, spot2, n2)
    plt.show()

    cv2.imshow("img", result_image)
    cv2.waitKey(0)


def f_func(b, image, spot1, n1, spot2, n2):
    sum1 = i_sum(image, spot1, b)   # верхняя часть дроби
    sum2 = i_sum(image, spot2, b)   # нижняя часть дроби
    return sum1 / sum2 - n1 / n2


def i_sum(image, spot, b):
    i = 0
    #   массив с интенсивностью всех пикселей в пятне
    intensity_list = intensityOfPixels(image, spot[0], spot[1], spot[2])
    for intensity in intensity_list:
        i += intensity ** (1 / b)
    return i


def numberOfPixels(r):
    return pi * r ** 2


#   определяем интенсивность пикселей в пятне вокруг кординаты с центром и радиусом
#   применяю формулу окружности для определения принадлежности координаты к окружности
def intensityOfPixels(image, x, y, radius):
    radius += 3  # cлегка увеличил радиус
    for i in range(x - radius, x + radius):
        for j in range(y - radius, y + radius):
            r = ((i - x) ** 2 + (j - y) ** 2) ** (1 / 2)
            if r <= radius:
                intensity = image[j, i]

                # чтобы не учитывать откровенно белые области вокруг
                if intensity < 240:
                    yield intensity


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process()
