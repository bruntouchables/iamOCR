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
        if len(points) < 4:
            points.append((x, y))
            # cv.rectangle(test_image, (x, y), (x + 3, y + 3), (0, 0, 0))
            if len(points) == 4:
                # reset mouse callback
                cv.setMouseCallback('Input', lambda *args: None)

                # process marked points
                p1, p2, p3, p4 = points
                new_width = p2[0] - p1[0]
                new_height = p3[1] - p1[1]
                new_points = [(0, 0), (new_width, 0), (0, new_height), (new_width, new_height)]
                transform_matrix = cv.getPerspectiveTransform(np.float32(points), np.float32(new_points))
                new_image = cv.warpPerspective(test_image, transform_matrix, (new_width, new_height))
                while True:
                    # show an image
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
