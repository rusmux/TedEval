import math

import numpy as np
from Polygon import Polygon


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    boxes = np.empty([1, 8], dtype=np.int64)
    boxes[0, 0] = int(points[0])
    boxes[0, 4] = int(points[1])
    boxes[0, 1] = int(points[2])
    boxes[0, 5] = int(points[3])
    boxes[0, 2] = int(points[4])
    boxes[0, 6] = int(points[5])
    boxes[0, 3] = int(points[6])
    boxes[0, 7] = int(points[7])
    point_mat = boxes[0].reshape([2, 4]).T
    return Polygon(point_mat)


def rectangle_to_polygon(rect):
    boxes = np.empty([1, 8], dtype="int32")
    boxes[0, 0] = int(rect.xmin)
    boxes[0, 4] = int(rect.ymin)
    boxes[0, 1] = int(rect.xmax)
    boxes[0, 5] = int(rect.ymin)
    boxes[0, 2] = int(rect.xmax)
    boxes[0, 6] = int(rect.ymax)
    boxes[0, 3] = int(rect.xmin)
    boxes[0, 7] = int(rect.ymax)
    point_mat = boxes[0].reshape([2, 4]).T
    return Polygon(point_mat)


def polygon_to_points(polygon):
    point_mat = []
    for p in polygon:
        for i in range(len(p)):
            point_mat.extend(p[i])
    return point_mat


def get_intersection(pD, pG):
    pInt = pD & pG
    return 0 if len(pInt) == 0 else pInt.area()


def compute_ap(conf_list, match_list, num_gt_care):
    AP = 0
    if len(conf_list) > 0:
        conf_list = np.array(conf_list)
        match_list = np.array(match_list)
        sorted_ind = np.argsort(-conf_list)
        conf_list = conf_list[sorted_ind]
        match_list = match_list[sorted_ind]
        correct = 0
        for n in range(len(conf_list)):
            match = match_list[n]
            if match:
                correct += 1
                AP += float(correct) / (n + 1)
        if num_gt_care > 0:
            AP /= num_gt_care
    return AP


def get_points_distance(point_1, point_2):
    x_dist = math.fabs(point_1[0] - point_2[0])
    y_dist = math.fabs(point_1[1] - point_2[1])
    return math.sqrt(x_dist * x_dist + y_dist * y_dist)


def diag(points):
    diag1 = get_points_distance((points[0], points[1]), (points[4], points[5]))
    diag2 = get_points_distance((points[2], points[3]), (points[6], points[7]))
    return (diag1 + diag2) / 2


def center_distance(p1, p2):
    return get_points_distance(p1.center(), p2.center())


def get_midpoints(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def get_angle_3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
    Returns a float between 0.0 and 360.0"""
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def gt_box_to_chars(num, points):
    chars = []
    if len(points) != 8:
        raise ValueError(f"points length must be 8, got {len(points)}")
    p1 = get_midpoints([points[0], points[1]], [points[6], points[7]])
    p2 = get_midpoints([points[2], points[3]], [points[4], points[5]])
    x_unit = (p2[0] - p1[0]) / num
    y_unit = (p2[1] - p1[1]) / num
    for i in range(num):
        x = p1[0] + x_unit / 2 + x_unit * i
        y = p1[1] + y_unit / 2 + y_unit * i
        chars.append((x, y))
    return chars
