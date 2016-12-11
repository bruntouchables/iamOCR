""" Created by Henrikh Kantuni on 12/4/16 """

import numpy as np
import cv2 as cv

# read an image
test_image = cv.imread('test.jpg', 0)

# keep marked points
points = []


def mark_point(event, x, y, *_):
    """
    Mark 4 points on the image
    1   2
    3   4
    to do the warp perspective transform

    :param event: mouse event
    :param x: x coordinate of the mouse event
    :param y: y coordinate of the mouse event
    :return: None
    """
    global test_image, points

    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) <= 4:
            cv.rectangle(test_image, (x, y), (x + 1, y + 1), (0, 0, 0))
            points.append((x, y))
            if len(points) == 4:
                # reset mouse callback
                cv.setMouseCallback('Input', lambda *args: None)

                # process marked points
                p1, p2, p3, p4 = points
                new_width = p2[0] - p1[0]
                new_height = p3[1] - p1[1]
                new_points = [(0, 0), (new_width, 0), (0, new_height), (new_width, new_height)]

                # warp perspective
                transform_matrix = cv.getPerspectiveTransform(np.float32(points), np.float32(new_points))
                new_image = cv.warpPerspective(test_image, transform_matrix, (new_width, new_height))

                # histogram equalization
                # new_image = cv.equalizeHist(new_image)

                # contrast limited adaptive histogram equalization
                # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # new_image = clahe.apply(new_image)

                # binary thresholding
                # thresh, new_image = cv.threshold(new_image, 127, 255, cv.THRESH_BINARY)

                # Otsu's thresholding
                thresh, new_image = cv.threshold(new_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                # morphological opening
                kernel = np.ones((13, 13), np.uint8)
                shapes = cv.morphologyEx(new_image, cv.MORPH_OPEN, kernel)

                # sharpening
                # kernel = np.array([
                #     [-1, -1, -1, -1, -1],
                #     [-1, 2, 2, 2, -1],
                #     [-1, 2, 8, 2, -1],
                #     [-1, 2, 2, 2, -1],
                #     [-1, -1, -1, -1, -1]
                # ]) / 8.0
                # new_image = cv.filter2D(new_image, -1, kernel)

                # apply automatic Canny edge detection using the computed median
                # sigma = 0.33
                # v = np.median(new_image)
                # lower = int(max(0, (1.0 - sigma) * v))
                # upper = int(min(255, (1.0 + sigma) * v))
                # new_image = cv.Canny(new_image, lower, upper)

                # find contours
                shapes, contours, hierarchy = cv.findContours(shapes, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                new_image = cv.cvtColor(new_image, cv.COLOR_GRAY2BGR)

                # find the largest contour
                largest_area_contour = sorted(contours, key=cv.contourArea)[-1]
                for index, contour in enumerate(contours):
                    # don't show the largest contour
                    if not np.array_equal(contour, largest_area_contour):
                        x, y, w, h = cv.boundingRect(contour)
                        cv.imshow('letter_{0}.jpg'.format(index), new_image[y:y+h, x:x+w])
                        cv.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

                while True:
                    # show an image
                    cv.namedWindow('Output', cv.WINDOW_KEEPRATIO)
                    cv.imshow('Output', new_image)
                    # exit on Esc
                    if cv.waitKey(20) & 0xFF == 27:
                        cv.destroyWindow('Output')
                        break


# capture mouse events
cv.namedWindow('Input')
cv.setMouseCallback('Input', mark_point)

while True:
    # show an image
    cv.imshow('Input', test_image)
    # exit on Esc
    if cv.waitKey(20) & 0xFF == 27:
        cv.destroyWindow('Input')
        break

# clean up
cv.destroyAllWindows()
