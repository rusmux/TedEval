import codecs
import re
import zipfile

import numpy as np


def load_zip_file(file, all_entries=False):
    archive = zipfile.ZipFile(file)

    pairs = []
    add_file = True
    for name in archive.namelist():
        keyName = name.replace("gt_", "").replace("res_", "").replace(".txt", "")

        if add_file:
            pairs.append([keyName, archive.read(name)])
        elif all_entries:
            raise ValueError(f"ZIP entry not valid: {name}")

    return dict(pairs)


def decode_utf8(raw):
    try:
        raw = codecs.decode(raw, "utf-8", "replace")
        # extracts BOM if exists
        raw = raw.encode("utf8")
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw.replace(codecs.BOM_UTF8, "", 1)
        return raw.decode("utf-8")
    except Exception:
        return None


def get_tl_line_values(line, LTRB=True, with_transcription=False, with_confidence=False, im_width=0, im_height=0):
    confidence = 0
    transcription = ""

    if LTRB:
        num_points = 4

        if with_transcription and with_confidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$",
                line,
            )
            if m is None:
                m = re.match(
                    r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$",
                    line,
                )
                raise ValueError(f'line format must be: "x_min,y_min,x_max,y_max,confidence,transcription", got {line}')
        elif with_confidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$",
                line,
            )
            if m is None:
                raise ValueError(f'line format must be: "x_min,y_min,x_max,y_max,confidence", got {line}')
        elif with_transcription:
            m = re.match(r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$", line)
            if m is None:
                raise ValueError(f'line format must be: "x_min,y_min,x_max,y_max,transcription", got {line}')
        else:
            m = re.match(r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$", line)
            if m is None:
                raise ValueError(f'line format must be: "x_min,y_min,x_max,y_max", got {line}')

        x_min = int(m.group(1))
        y_min = int(m.group(2))
        x_max = int(m.group(3))
        y_max = int(m.group(4))
        if x_max < x_min:
            raise ValueError(f"x_max ({x_max}) must be greater or equal to x_min ({x_min})")
        if y_max < y_min:
            raise ValueError(f"y_max ({y_max}) must be greater or equal to y_min ({y_min})")

        points = [float(m.group(i)) for i in range(1, (num_points + 1))]

        if im_width > 0 and im_height > 0:
            validate_point_inside_bounds(x_min, y_min, im_width, im_height)
            validate_point_inside_bounds(x_max, y_max, im_width, im_height)

    else:
        num_points = 8

        if with_transcription and with_confidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$",
                line,
            )
            if m is None:
                raise ValueError(f'line format must be: "x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription", got {line}')
        elif with_confidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$",
                line,
            )
            if m is None:
                raise ValueError(f'line format must be: "x1,y1,x2,y2,x3,y3,x4,y4,confidence", got {line}')
        elif with_transcription:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$",
                line,
            )
            if m is None:
                raise ValueError(f'line format must be: "x1,y1,x2,y2,x3,y3,x4,y4,transcription", got {line}')
        else:
            if line[-1] == ",":
                line = line[:-1]
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$",
                line,
            )
            if m is None:
                raise ValueError(f'line format must be: "x1,y1,x2,y2,x3,y3,x4,y4", got {line}')

        points = [float(m.group(i)) for i in range(1, (num_points + 1))]

        validate_clockwise_points(points)

        if im_width > 0 and im_height > 0:
            validate_point_inside_bounds(points[0], points[1], im_width, im_height)
            validate_point_inside_bounds(points[2], points[3], im_width, im_height)
            validate_point_inside_bounds(points[4], points[5], im_width, im_height)
            validate_point_inside_bounds(points[6], points[7], im_width, im_height)

    if with_confidence:
        confidence = float(m.group(num_points + 1))

    if with_transcription:
        pos_transcription = num_points + (2 if with_confidence else 1)
        transcription = m.group(pos_transcription)
        m2 = re.match(r"^\s*\"(.*)\"\s*$", transcription)
        if m2 is not None:  # Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2[1].replace("\\\\", "\\").replace('\\"', '"')

    return points, confidence, transcription


def validate_point_inside_bounds(x, y, im_width, im_height):
    if x < 0 or x > im_width:
        raise ValueError(f"x value ({x}) is not valid. Image dimensions: ({im_width},{im_height})")
    if y < 0 or y > im_height:
        raise ValueError(f"y value ({y})  not valid. Image dimensions: ({im_width},{im_height})")


def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 8:
        raise ValueError(f"points length must be 8, got {len(points)}")

    point = [
        [int(points[0]), int(points[1])],
        [int(points[2]), int(points[3])],
        [int(points[4]), int(points[5])],
        [int(points[6]), int(points[7])],
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1]),
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3]
    if summatory > 0:
        raise ValueError(
            "points are not ordered clockwise. The coordinates of bounding quadrilaterals have to be given in "
            "clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate "
            "system used is the standard one, with the image origin at the upper left, the X axis extending to the "
            "right and Y axis extending downwards.",
        )


def get_tl_line_values_from_file_contents(
    content,
    CRLF=True,
    LTRB=True,
    with_transcription=False,
    with_confidence=False,
    im_width=0,
    im_height=0,
    sort_by_confidences=True,
):
    """
    Returns all points, confidences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    points_list = []
    transcriptions_list = []
    confidences_list = []

    lines = content.split("\r\n" if CRLF else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line != "":
            points, confidence, transcription = get_tl_line_values(
                line,
                LTRB,
                with_transcription,
                with_confidence,
                im_width,
                im_height,
            )
            points_list.append(points)
            transcriptions_list.append(transcription)
            confidences_list.append(confidence)

    if with_confidence and confidences_list and sort_by_confidences:
        sorted_ind = np.argsort(-np.array(confidences_list))
        confidences_list = [confidences_list[i] for i in sorted_ind]
        points_list = [points_list[i] for i in sorted_ind]
        transcriptions_list = [transcriptions_list[i] for i in sorted_ind]

    return points_list, confidences_list, transcriptions_list
