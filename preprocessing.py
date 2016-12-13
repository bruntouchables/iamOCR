""" Created by Henrikh Kantuni on 12/4/16 """

import numpy as np
import cv2


def find_words(image):
    """
    Find and outline words in the given image

    :param image: numpy array
    :return: None
    """
    # downsample
    small = cv2.pyrDown(image)

    # add color to an image to make ROI (region of interest) areas green
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

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
        ratio = cv2.countNonZero(mask_roi) / w * h
        print(ratio)

        if ratio > .45 and h > 8 and w > 8:
            # don't forget to upsample the rectangle
            cv2.rectangle(image, (2 * x, 2 * y), (2 * (x + w), 2 * (y + h)), (0, 255, 0), 1)

    return image


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

                # find words
                new_image = find_words(new_image)

                # histogram equalization
                # new_image = cv2.equalizeHist(new_image)

                # contrast limited adaptive histogram equalization
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # new_image = clahe.apply(new_image)

                # binary threshold
                # thresh, new_image = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY)

                # Otsu's threshold
                # thresh, new_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # morphological opening
                # kernel = np.ones((13, 13), np.uint8)
                # shapes = cv2.morphologyEx(new_image, cv2.MORPH_OPEN, kernel)

                # sharpen image
                # kernel = np.array([
                #     [-1, -1, -1, -1, -1],
                #     [-1, 2, 2, 2, -1],
                #     [-1, 2, 8, 2, -1],
                #     [-1, 2, 2, 2, -1],
                #     [-1, -1, -1, -1, -1]
                # ]) / 8.0
                # new_image = cv2.filter2D(new_image, -1, kernel)

                # apply automatic Canny edge detection using the computed median
                # sigma = 0.33
                # v = np.median(new_image)
                # lower = int(max(0, (1.0 - sigma) * v))
                # upper = int(min(255, (1.0 + sigma) * v))
                # new_image = cv2.Canny(new_image, lower, upper)

                # find contours
                # shapes, contours, hierarchy = cv2.findContours(shapes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)

                # find the largest contour
                # if len(contours) > 0:
                #     largest_area_contour = sorted(contours, key=cv2.contourArea)[-1]
                #     for index, contour in enumerate(contours):
                #         # don't show the largest contour
                #         if not np.array_equal(contour, largest_area_contour):
                #             x, y, w, h = cv2.boundingRect(contour)
                #             paragraph = new_image[y:y + h, x:x + w]
                #             # sharpening
                #             kernel = np.array([
                #                 [-1, -1, -1, -1, -1],
                #                 [-1, 2, 2, 2, -1],
                #                 [-1, 2, 8, 2, -1],
                #                 [-1, 2, 2, 2, -1],
                #                 [-1, -1, -1, -1, -1]
                #             ]) / 8.0
                #             paragraph = cv2.filter2D(paragraph, -1, kernel)
                #             cv2.imshow('letter_{0}.jpg'.format(index), paragraph)
                #             cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

                while True:
                    # show an image
                    cv2.namedWindow('Output')
                    cv2.imshow('Output', new_image)
                    # exit on Esc
                    if cv2.waitKey(20) & 0xFF == 27:
                        cv2.destroyWindow('Output')
                        break


# read an image
test_image = cv2.imread('test.jpg', 0)

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
