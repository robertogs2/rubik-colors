import numpy as np
import operator
import cv2
import math
from cv2 import *

aspect_ratio_lower = 0.5
aspect_ratio_higher = 1.4
area_limit_higher = 9500
area_limit_lower = 2200
show_thresh = False
paint_all = False
distance_threshold = 20

orange_bounds = {"min": np.array([5, 95, 100]), "max": np.array([22, 255, 255])}
blue_bounds = {"min": np.array([95, 80, 110]), "max": np.array([130, 255, 255])}
white_bounds = {"min": np.array([0, 0, 180]), "max": np.array([180, 63, 255])}
yellow_bounds = {"min": np.array([22, 50, 130]), "max": np.array([43, 255, 255])}
green_bounds = {"min": np.array([40, 75, 80]), "max": np.array([85, 255, 255])}

red_bounds = {"min": np.array([0, 175, 150]), "max": np.array([5, 255, 255])}
red_bounds2 = {"min": np.array([155, 80, 90]), "max": np.array([180, 255, 255])}

colors_bounds = [orange_bounds, blue_bounds, red_bounds, white_bounds, yellow_bounds, green_bounds]
colors = ["O", "B", "R", "W", "Y", "G"]
instructions = ["Show your front", "Rotate to your right",
                "Rotate to your right", "Rotate to your right",
                "Rotate to your right to return to front and rotate frontwards",
                "Rotate backwards to return to front and rotate backwards"]


def find_contours(hsv_image, color):
    color_index = colors.index(color)
    color_min = colors_bounds[color_index]["min"]
    color_max = colors_bounds[color_index]["max"]

    frame_threshed = cv2.inRange(hsv_image, color_min, color_max)  # Thresholding imageq
    if color == 'R':
        color_min = red_bounds2["min"]
        color_max = red_bounds2["max"]
        frame_threshed2 = cv2.inRange(hsv_image, color_min, color_max)  # Thresholding imageq
        frame_threshed = bitwise_or(frame_threshed, frame_threshed2)
    if show_thresh:
        imshow(color, frame_threshed)

    ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Filters contours into coordinates
def filter_contours(contours, color):
    coordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = h * w
        aspect_ratio = float(w) / h
        if aspect_ratio_higher > abs(aspect_ratio) > aspect_ratio_lower:
            if area_limit_higher > abs(area) > area_limit_lower:
                coordinates += [(x, y, color)]
    return coordinates


# Paints a given set of colors in an image
def paint_contours(im, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = h * w
        aspect_ratio = float(w) / h
        #print(aspect_ratio)
        if paint_all or (aspect_ratio_higher > abs(aspect_ratio) > aspect_ratio_lower):
            if paint_all or (area_limit_higher > abs(area) > area_limit_lower):
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Sort different coordinates into a matrix
def sort_coordinates(coordinates):
    print("Found: " + str(len(coordinates)))
    if len(coordinates) == 9:
        print("Sorting 9")
        coordinates.sort(key=operator.itemgetter(1))
        line0 = coordinates[0:3]
        line1 = coordinates[3:6]
        line2 = coordinates[6:9]
        line0.sort(key=operator.itemgetter(0))
        line1.sort(key=operator.itemgetter(0))
        line2.sort(key=operator.itemgetter(0))
        result = line0 + line1 + line2
        return result
    elif len(coordinates) == 8:
        print("Sorting 8")
        coordinates.sort(key=operator.itemgetter(1))
        line0 = coordinates[0:3]
        line1 = coordinates[3:5]
        line2 = coordinates[5:8]
        line0.sort(key=operator.itemgetter(0))
        line1.sort(key=operator.itemgetter(0))
        line2.sort(key=operator.itemgetter(0))
        result = line0 + line1 + line2
        return result
    else:
        return coordinates


# Filters only the color letter from matrix
def filter_color_letter(coordinates):
    result = []
    if len(coordinates) > 0:
        for l in coordinates:
            result += [l[2]]
    return result


# Filters coordinates to not be close
def filter_distance(coordinates):
    for coord in coordinates:
        for anotherCoord in coordinates:
            if coord != anotherCoord:
                dx = coord[0] - anotherCoord[0]
                dy = coord[1] - anotherCoord[1]
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < distance_threshold:
                    coordinates.remove(anotherCoord)
    return coordinates


# Gets matrix of color
def get_color_matrix(im):
    im = cv2.bilateralFilter(im, 9, 75, 75)
    im = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    coordinates = []
    for color in colors:
        contours = find_contours(hsv_image, color)
        color_coordinates = filter_contours(contours, color)
        coordinates += color_coordinates
        paint_contours(im, contours)
    cv2.imshow("Result", im)

    coordinates = filter_distance(coordinates)
    coordinates_sort = sort_coordinates(coordinates)
    letters_sort = filter_color_letter(coordinates_sort)
    return coordinates_sort, letters_sort


# Parses list of strings to a single string
def color_list2str(l):
    if len(l) == 8:
        result = ''
        for s in l:
            result += s
        return result[0:4] + 'X' + result[4:8]
    else:
        result = ''
        for s in l:
            result += s
        return result


def get_color_strings():
    dim = (640, 480)
    cap = cv2.VideoCapture(0)
    w = h = 300
    x = 170
    y = 90
    results = []
    running = True
    print(instructions[0])
    while running:
        ret, image = cap.read()
        image = cv2.resize(image, dim)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Camera feed", image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            crop_image = image[y:y + h, x:x + w]
            coordinates, letters = get_color_matrix(crop_image)
            print(letters)
            # Choose if it works
            k = cv2.waitKey() & 0xFF
            if k == ord(' '):
                string = color_list2str(letters)
                print('Added selected frame as ' + string + '\n')
                results += [string]
                cv2.destroyAllWindows()
                if len(results) == 6:  # got all the faces
                    print("All faces taken")
                    break
                print(instructions[len(results)])
        elif k == ord('q'):
            running = False

    print("Finish")
    cap.release()
    cv2.destroyAllWindows()
    return results
