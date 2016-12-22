""" Created by Henrikh Kantuni on 12/4/16 """

import numpy as np
import cv2


def detect_text(image):
    """
    Detect text in the given image

    :param image: numpy array
    :return: array of text areas as [y, y + h, x, x + w] in the image
    """
    text_areas = []

    # downsample
    small = cv2.pyrDown(image)

    # morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    # binary threshold
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # connect horizontally oriented regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # find contours
    mask = np.zeros(binary.shape)
    connected, contours, hierarchy = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        mask_roi = mask[y: y + h, x: x + w]

        # fill the contour
        cv2.drawContours(mask, contours, index, (255, 255, 255), -1)

        # calculate a ratio of non-zero pixels in the filled region
        ratio = cv2.countNonZero(mask_roi) / (w * h)

        if ratio > 0.5 and h > 8 and w > 8:
            # don't forget to upsample the rectangle
            text_areas.append([2 * y, 2 * (y + h), 2 * x, 2 * (x + w)])

    return text_areas


def mark_points(event, x, y, *_):
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

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) <= 4:
            cv2.rectangle(test_image, (x, y), (x + 1, y + 1), (255, 255, 255))
            points.append((x, y))
            if len(points) == 4:
                # reset mouse callback
                cv2.setMouseCallback('Input', lambda *args: None)

                # process marked points
                p1, p2, p3, p4 = points
                new_width = p2[0] - p1[0]
                new_height = p3[1] - p1[1]
                new_points = [(0, 0), (new_width, 0), (0, new_height), (new_width, new_height)]

                # warp perspective transform
                kernel = cv2.getPerspectiveTransform(np.float32(points), np.float32(new_points))
                new_image = cv2.warpPerspective(test_image, kernel, (new_width, new_height))

                # find text in the image
                text_areas = detect_text(new_image)

                # add color to the image to make ROI (region of interest) areas green
                new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)

                for index, area in enumerate(text_areas):
                    # area format: [y, y + h, x, x + w]
                    cv2.rectangle(new_image, (area[2], area[0]), (area[3], area[1]), (0, 255, 0), 1)

                while True:
                    # show an image
                    cv2.namedWindow('Output')
                    cv2.imshow('Output', new_image)
                    # exit on Esc
                    if cv2.waitKey(20) & 0xFF == 27:
                        cv2.destroyWindow('Output')
                        break


# read an image
test_image = cv2.imread('test_1.jpg', 0)

# keep marked points
points = []

# capture mouse events
cv2.namedWindow('Input')
cv2.setMouseCallback('Input', mark_points)

while True:
    # show an image
    cv2.imshow('Input', test_image)
    # exit on Esc
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.destroyWindow('Input')
        break

# clean up
cv2.destroyAllWindows()
