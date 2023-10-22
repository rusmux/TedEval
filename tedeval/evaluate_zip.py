import typing as tp
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tedeval.config import EVALUATION_PARAMS
from tedeval.utils.points import (
    center_distance,
    compute_ap,
    diag,
    get_angle_3pt,
    get_intersection,
    get_midpoints,
    gt_box_to_chars,
    polygon_from_points,
    polygon_to_points,
    rectangle_to_polygon,
)
from tedeval.utils.zip import decode_utf8, get_tl_line_values_from_file_contents, load_zip_file


def evaluate_zip_text_detections(
    ground_truth_file: tp.Union[Path, tp.IO[bytes]],
    predictions_file: tp.Union[Path, tp.IO[bytes]],
    evaluation_params: dict = EVALUATION_PARAMS,
) -> tp.Dict[str, dict]:
    def char_fill(det_nums: Sequence[int], match_mat: np.ndarray) -> None:
        for det_num in det_nums:
            det_pol = det_pols[det_num]
            for gt_num, gt_chars in enumerate(gt_char_points):
                if match_mat[gt_num, det_num] == 1:
                    for gt_char_num, gtChar in enumerate(gt_chars):
                        if det_pol.isInside(gtChar[0], gtChar[1]):
                            gt_char_counts[gt_num][det_num][gt_char_num] = 1

    def one_to_one_match(row: int, col: int) -> bool:
        if not (
            recall_mat[row, col] >= evaluation_params["RECALL_IOU_THRESHOLD"]
            and precision_mat[row, col] >= evaluation_params["PRECISION_IOU_THRESHOLD"]
        ):
            return False

        cont = 0
        for j in range(len(recall_mat[0])):
            if (
                recall_mat[row, j] >= evaluation_params["RECALL_IOU_THRESHOLD"]
                and precision_mat[row, j] >= evaluation_params["PRECISION_IOU_THRESHOLD"]
            ):
                cont += 1
        if cont != 1:
            return False
        cont = 0
        for i in range(len(recall_mat)):
            if (
                recall_mat[i, col] >= evaluation_params["RECALL_IOU_THRESHOLD"]
                and precision_mat[i, col] >= evaluation_params["PRECISION_IOU_THRESHOLD"]
            ):
                cont += 1
        return cont == 1

    def one_to_many_match(gt_num: int) -> tuple[bool, list[int]]:
        many_sum = 0
        det_rects = []
        for det_num in range(len(recall_mat[0])):
            if (
                det_num not in det_dont_care_pols_num
                and gt_exclude_mat[gt_num] == 0
                and det_exclude_mat[det_num] == 0
                and precision_mat[gt_num, det_num] >= evaluation_params["PRECISION_IOU_THRESHOLD"]
            ):
                many_sum += recall_mat[gt_num, det_num]
                det_rects.append(det_num)
        if many_sum >= evaluation_params["RECALL_IOU_THRESHOLD"] and len(det_rects) >= 2:
            pivots = []
            for matchDet in det_rects:
                pD = polygon_from_points(det_pol_points[matchDet])
                pivots.append([get_midpoints(pD[0][0], pD[0][3]), pD.center()])
            for i in range(len(pivots)):
                for k in range(len(pivots)):
                    if k == i:
                        continue
                    angle = get_angle_3pt(pivots[i][0], pivots[k][1], pivots[i][1])
                    if angle > 180:
                        angle = 360 - angle
                    if min(angle, 180 - angle) >= 45:
                        return False, []
            return True, det_rects
        else:
            return False, []

    def many_to_one_match(det_num: int) -> tuple[bool, list[int]]:
        many_sum = 0
        gt_rects = []
        for gt_num in range(len(recall_mat)):
            if (
                gt_num not in gt_dont_care_pols_num
                and gt_exclude_mat[gt_num] == 0
                and det_exclude_mat[det_num] == 0
                and recall_mat[gt_num, det_num] >= evaluation_params["RECALL_IOU_THRESHOLD"]
            ):
                many_sum += precision_mat[gt_num, det_num]
                gt_rects.append(gt_num)
        if many_sum >= evaluation_params["PRECISION_IOU_THRESHOLD"] and len(gt_rects) >= 2:
            pivots = []
            for matchGt in gt_rects:
                pG = gt_pols[matchGt]
                pivots.append([get_midpoints(pG[0][0], pG[0][3]), pG.center()])
            for i in range(len(pivots)):
                for k in range(len(pivots)):
                    if k == i:
                        continue
                    angle = get_angle_3pt(pivots[i][0], pivots[k][1], pivots[i][1])
                    if angle > 180:
                        angle = 360 - angle
                    if min(angle, 180 - angle) >= 45:
                        return False, []
            return True, gt_rects
        else:
            return False, []

    per_sample_metrics = {}

    method_recall_sum = 0
    method_precision_sum = 0

    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

    gt = load_zip_file(ground_truth_file)
    subm = load_zip_file(predictions_file, True)

    num_global_care_gt = 0
    num_global_care_det = 0

    arr_global_confidences = []
    arr_global_matches = []

    for res_file in gt:
        gt_file = decode_utf8(gt[res_file])

        recall_accum = 0
        precision_accum = 0

        recall_mat = np.empty([1, 1])
        precision_mat = np.empty([1, 1])

        gt_pols = []
        det_pols = []

        gt_pol_points = []
        det_pol_points = []

        # pseudo character centers
        gt_char_points = []
        gt_char_counts = []

        # visualization
        char_counts = np.zeros([1, 1])
        recall_score = []
        precision_score = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gt_dont_care_pols_num = []
        # Array of Detected Polygons' matched with a don't Care GT
        det_dont_care_pols_num = []

        pairs = []
        det_matched_nums = []

        arr_sample_confidences = []
        arr_sample_match = []
        sample_ap = 0

        evaluation_log = ""

        points_list, _, transcriptions_list = get_tl_line_values_from_file_contents(
            gt_file,
            evaluation_params["GT_CRLF"],
            evaluation_params["GT_LTRB"],
            True,
        )
        for n in range(len(points_list)):
            points = points_list[n]
            transcription = transcriptions_list[n]
            dont_care = transcription == "###"
            if evaluation_params["GT_LTRB"]:
                gt_rect = Rectangle(*points)
                gt_pol = rectangle_to_polygon(gt_rect)
                points = polygon_to_points(gt_pol)
            else:
                gt_pol = polygon_from_points(points)
            gt_pols.append(gt_pol)
            if dont_care:
                gt_dont_care_pols_num.append(len(gt_pols) - 1)
                gt_pol_points.append(points)
                gt_char_points.append([])
            else:
                gt_char_size = len(transcription)
                aspect_ratio = gt_pol.aspectRatio()
                if aspect_ratio > 1.5:
                    points_ver = [
                        points[6],
                        points[7],
                        points[0],
                        points[1],
                        points[2],
                        points[3],
                        points[4],
                        points[5],
                    ]
                    gt_pol_points.append(points_ver)
                    gt_char_points.append(gt_box_to_chars(gt_char_size, points_ver))
                else:
                    gt_char_points.append(gt_box_to_chars(gt_char_size, points))
                    gt_pol_points.append(points)
        evaluation_log += (
            "GT polygons: "
            + str(len(gt_pols))
            + (f" ({len(gt_dont_care_pols_num)}" + " don't care)\n" if gt_dont_care_pols_num else "\n")
        )

        # GT Don't Care overlap
        for DontCare in gt_dont_care_pols_num:
            for gtNum in list(set(range(len(gt_pols))) - set(gt_dont_care_pols_num)):
                if get_intersection(gt_pols[gtNum], gt_pols[DontCare]) > 0:
                    gt_pols[DontCare] -= gt_pols[gtNum]

        if res_file in subm:
            det_file = decode_utf8(subm[res_file])

            points_list, confidences_list, _ = get_tl_line_values_from_file_contents(
                det_file,
                evaluation_params["DET_CRLF"],
                evaluation_params["DET_LTRB"],
                evaluation_params["TRANSCRIPTION"],
                evaluation_params["CONFIDENCES"],
            )
            for n in range(len(points_list)):
                points = points_list[n]

                if evaluation_params["DET_LTRB"]:
                    det_rect = Rectangle(*points)
                    det_pol = rectangle_to_polygon(det_rect)
                    points = polygon_to_points(det_pol)
                else:
                    det_pol = polygon_from_points(points)
                det_pols.append(det_pol)
                det_pol_points.append(points)

            evaluation_log += f"DET polygons: {len(det_pols)}"

            if gt_pols and det_pols:
                # Calculate IoU and precision matrix
                output_shape = [len(gt_pols), len(det_pols)]
                recall_mat = np.empty(output_shape)
                precision_mat = np.empty(output_shape)
                match_mat = np.zeros(output_shape)
                gt_rect_mat = np.zeros(len(gt_pols), np.int8)
                det_rect_mat = np.zeros(len(det_pols), np.int8)
                gt_exclude_mat = np.zeros(len(gt_pols), np.int8)
                det_exclude_mat = np.zeros(len(det_pols), np.int8)
                for gtNum in range(len(gt_pols)):
                    det_char_counts = []
                    for detNum in range(len(det_pols)):
                        pG = gt_pols[gtNum]
                        pD = det_pols[detNum]
                        intersected_area = get_intersection(pD, pG)
                        recall_mat[gtNum, detNum] = 0 if pG.area() == 0 else intersected_area / pG.area()
                        precision_mat[gtNum, detNum] = 0 if pD.area() == 0 else intersected_area / pD.area()
                        det_char_counts.append(np.zeros(len(gt_char_points[gtNum])))
                    gt_char_counts.append(det_char_counts)

                # Find detection Don't Care
                if gt_dont_care_pols_num:
                    for detNum in range(len(det_pols)):
                        # many-to-one
                        many_sum = 0
                        for gtNum in gt_dont_care_pols_num:
                            if recall_mat[gtNum, detNum] > evaluation_params["RECALL_IOU_THRESHOLD"]:
                                many_sum += precision_mat[gtNum, detNum]
                        if many_sum >= evaluation_params["PRECISION_IOU_THRESHOLD"]:
                            det_dont_care_pols_num.append(detNum)
                        else:
                            for gtNum in gt_dont_care_pols_num:
                                if precision_mat[gtNum, detNum] > evaluation_params["PRECISION_IOU_THRESHOLD"]:
                                    det_dont_care_pols_num.append(detNum)
                                    break
                        # many-to-one for mixed DC and non-DC
                        for gtNum in gt_dont_care_pols_num:
                            if recall_mat[gtNum, detNum] > 0:
                                det_pols[detNum] -= gt_pols[gtNum]

                    evaluation_log += (
                        f" ({len(det_dont_care_pols_num)}" + " don't care)\n" if det_dont_care_pols_num else "\n"
                    )

                # Recalculate matrices
                for gtNum in range(len(gt_pols)):
                    for detNum in range(len(det_pols)):
                        pG = gt_pols[gtNum]
                        pD = det_pols[detNum]
                        intersected_area = get_intersection(pD, pG)
                        recall_mat[gtNum, detNum] = 0 if pG.area() == 0 else intersected_area / pG.area()
                        precision_mat[gtNum, detNum] = 0 if pD.area() == 0 else intersected_area / pD.area()

                # Find many-to-one matches
                evaluation_log += "Find many-to-one matches\n"
                for detNum in range(len(det_pols)):
                    if detNum not in det_dont_care_pols_num:
                        match, matches_gt = many_to_one_match(detNum)
                        if match:
                            pairs.append({"gt": matches_gt, "det": [detNum], "type": "MO"})
                            evaluation_log += f"Match GT #{str(matches_gt)} with Det #{str(detNum)}" + "\n"

                # Find one-to-one matches
                evaluation_log += "Find one-to-one matches\n"
                for gtNum in range(len(gt_pols)):
                    for detNum in range(len(det_pols)):
                        if gtNum not in gt_dont_care_pols_num and detNum not in det_dont_care_pols_num:
                            match = one_to_one_match(gtNum, detNum)
                            if match:
                                norm_dist = center_distance(gt_pols[gtNum], det_pols[detNum])
                                norm_dist /= diag(gt_pol_points[gtNum]) + diag(det_pol_points[detNum])
                                norm_dist *= 2
                                if norm_dist < evaluation_params["EV_PARAM_IND_CENTER_DIFF_THR"]:
                                    pairs.append({"gt": [gtNum], "det": [detNum], "type": "OO"})
                                    evaluation_log += f"Match GT #{str(gtNum)} with Det #{str(detNum)}" + "\n"

                # Find one-to-many matches
                evaluation_log += "Find one-to-many matches\n"
                for gtNum in range(len(gt_pols)):
                    if gtNum not in gt_dont_care_pols_num:
                        match, matches_det = one_to_many_match(gtNum)
                        if match:
                            pairs.append({"gt": [gtNum], "det": matches_det, "type": "OM"})
                            evaluation_log += f"Match Gt #{str(gtNum)} with Det #{str(matches_det)}" + "\n"

                # Fill match matrix
                for pair in pairs:
                    match_mat[pair["gt"], pair["det"]] = 1

                # Fill character matrix
                char_fill(np.where(match_mat.sum(axis=0) > 0)[0], match_mat)

                # Recall score
                for gtNum in range(len(gt_rect_mat)):
                    if match_mat.sum(axis=1)[gtNum] > 0:
                        recall_accum += len(np.where(sum(gt_char_counts[gtNum]) == 1)[0]) / len(gt_char_points[gtNum])
                        if len(np.where(sum(gt_char_counts[gtNum]) == 1)[0]) / len(gt_char_points[gtNum]) < 1:
                            recall_score.append(
                                "<font color=red>"
                                + str(len(np.where(sum(gt_char_counts[gtNum]) == 1)[0]))
                                + "/"
                                + str(len(gt_char_points[gtNum]))
                                + "</font>",
                            )
                        else:
                            recall_score.append(
                                str(len(np.where(sum(gt_char_counts[gtNum]) == 1)[0]))
                                + "/"
                                + str(len(gt_char_points[gtNum])),
                            )
                    else:
                        recall_score.append("")

                # Precision score
                for detNum in range(len(det_rect_mat)):
                    if match_mat.sum(axis=0)[detNum] > 0:
                        det_total = 0
                        det_contain = 0
                        for gtNum in range(len(gt_rect_mat)):
                            if match_mat[gtNum, detNum] > 0:
                                det_total += len(gt_char_counts[gtNum][detNum])
                                det_contain += len(np.where(gt_char_counts[gtNum][detNum] == 1)[0])
                        precision_accum += det_contain / det_total
                        if det_contain / det_total < 1:
                            precision_score.append(f"<font color=red>{str(det_contain)}/{str(det_total)}</font>")
                        else:
                            precision_score.append(f"{str(det_contain)}/{str(det_total)}")
                    else:
                        precision_score.append("")

                # Visualization
                char_counts = np.zeros((len(gt_rect_mat), len(det_rect_mat)))
                for gtNum in range(len(gt_rect_mat)):
                    for detNum in range(len(det_rect_mat)):
                        char_counts[gtNum][detNum] = sum(gt_char_counts[gtNum][detNum])

            if evaluation_params["CONFIDENCES"]:
                for detNum in range(len(det_pols)):
                    if detNum not in det_dont_care_pols_num:
                        match = detNum in det_matched_nums
                        arr_sample_confidences.append(confidences_list[detNum])
                        arr_sample_match.append(match)
                        arr_global_confidences.append(confidences_list[detNum])
                        arr_global_matches.append(match)

        num_gt_care = len(gt_pols) - len(gt_dont_care_pols_num)
        num_det_care = len(det_pols) - len(det_dont_care_pols_num)
        if num_gt_care == 0:
            recall = float(1)
            precision = float(0) if num_det_care > 0 else float(1)
            sample_ap = precision
        else:
            recall = float(recall_accum) / num_gt_care
            precision = float(0) if num_det_care == 0 else float(precision_accum) / num_det_care
            if evaluation_params["CONFIDENCES"] and evaluation_params["PER_SAMPLE_RESULTS"]:
                sample_ap = compute_ap(arr_sample_confidences, arr_sample_match, num_gt_care)

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        evaluation_log += (
            "<b>Recall = "
            + str(round(recall_accum, 2))
            + " / "
            + str(num_gt_care)
            + " = "
            + str(round(recall, 2))
            + "\n</b>"
        )
        evaluation_log += (
            "<b>Precision = "
            + str(round(precision_accum, 2))
            + " / "
            + str(num_det_care)
            + " = "
            + str(round(precision, 2))
            + "\n</b>"
        )

        method_recall_sum += recall_accum
        method_precision_sum += precision_accum
        num_global_care_gt += num_gt_care
        num_global_care_det += num_det_care

        if evaluation_params["PER_SAMPLE_RESULTS"]:
            per_sample_metrics[res_file] = {
                "precision": precision,
                "recall": recall,
                "hmean": hmean,
                "pairs": pairs,
                "AP": sample_ap,
                "recall_mat": [] if len(det_pols) > 100 else recall_mat.tolist(),
                "precision_mat": [] if len(det_pols) > 100 else precision_mat.tolist(),
                "gt_pol_points": gt_pol_points,
                "det_pol_points": det_pol_points,
                "gt_char_points": gt_char_points,
                "gt_char_counts": [sum(k).tolist() for k in gt_char_counts],
                "char_counts": char_counts.tolist(),
                "recall_score": recall_score,
                "precision_score": precision_score,
                "gtDontCare": gt_dont_care_pols_num,
                "detDontCare": det_dont_care_pols_num,
                "evaluationParams": evaluation_params,
                "evaluation_log": evaluation_log,
            }

    # Compute MAP and MAR
    AP = 0
    if evaluation_params["CONFIDENCES"]:
        AP = compute_ap(arr_global_confidences, arr_global_matches, num_global_care_gt)

    method_recall = 0 if num_global_care_gt == 0 else method_recall_sum / num_global_care_gt
    method_precision = 0 if num_global_care_det == 0 else method_precision_sum / num_global_care_det
    method_hmean = (
        0
        if method_recall + method_precision == 0
        else 2 * method_recall * method_precision / (method_recall + method_precision)
    )

    total_metrics = {"recall": method_recall, "precision": method_precision, "hmean": method_hmean, "AP": AP}

    return {"total_metrics": total_metrics, "per_sample_metrics": per_sample_metrics}
