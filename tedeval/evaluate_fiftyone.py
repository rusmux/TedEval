import io
import typing as tp
import zipfile

import fiftyone as fo
from tedeval.config import EVALUATION_PARAMS
from tedeval.evaluate_zip import evaluate_zip_text_detections


def polyline_to_points(  # noqa: WPS210
    polyline: fo.Polyline,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    if len(polyline["points"][0]) == 4:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = polyline["points"][0]  # noqa: WPS221
    else:
        detection = polyline.to_detection()
        x_min, y_min, width, height = detection["bounding_box"]
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = (
            (x_min, y_min),
            (x_min + width, y_min),
            (x_min + width, y_min + height),
            (x_min, y_min + height),
        )

    x1, x2, x3, x4 = (int(round(x_coord * image_width)) for x_coord in [x1, x2, x3, x4])  # noqa: WPS221
    y1, y2, y3, y4 = (int(round(y_coord * image_height)) for y_coord in [y1, y2, y3, y4])  # noqa: WPS221

    return x1, y1, x2, y2, x3, y3, x4, y4  # noqa: WPS227


def create_tedeval_labels(  # noqa: WPS210
    dataset: fo.Dataset,
    pred_field: str,
    gt_field: str,
) -> tuple[io.BytesIO, io.BytesIO, bool, bool]:
    ground_truth_zip_buffer = io.BytesIO()
    prediction_zip_buffer = io.BytesIO()

    has_transcription = False
    has_confidence = False

    for sample in dataset.iter_samples(progress=True):
        image_height = sample["metadata"]["height"]
        image_width = sample["metadata"]["width"]

        ground_truth_field = sample[gt_field]
        if isinstance(ground_truth_field, fo.Detections):
            ground_truth_field = ground_truth_field.to_polylines()

        with zipfile.ZipFile(ground_truth_zip_buffer, "a") as ground_truth_zip:
            ground_truth_label = ""
            for polyline in ground_truth_field["polylines"]:
                x1, y1, x2, y2, x3, y3, x4, y4 = polyline_to_points(  # noqa: WPS236
                    polyline,
                    image_width,
                    image_height,
                )
                ground_truth_label += f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4}"  # noqa: WPS221
                if polyline["label"]:
                    ground_truth_label += f",{polyline['label']}"
                ground_truth_label += "\n"
            ground_truth_zip.writestr(f"{sample.filepath}.txt", ground_truth_label)

        prediction_field = sample[pred_field]
        if isinstance(prediction_field, fo.Detections):
            prediction_field = prediction_field.to_polylines()

        with zipfile.ZipFile(prediction_zip_buffer, "a") as prediction_zip:
            prediction_label = ""
            for polyline in prediction_field["polylines"]:
                x1, y1, x2, y2, x3, y3, x4, y4 = polyline_to_points(polyline, image_width, image_height)  # noqa: WPS236
                prediction_label += f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4}"  # noqa: WPS221
                if polyline["confidence"]:
                    prediction_label += f",{polyline['confidence']}"
                    has_confidence = True
                if polyline["label"]:
                    prediction_label += f",{polyline['label']}"
                    has_transcription = True
                prediction_label += "\n"
            prediction_zip.writestr(f"{sample.filepath}.txt", prediction_label)

    return ground_truth_zip_buffer, prediction_zip_buffer, has_transcription, has_confidence


def evaluate_fiftyone_text_detections(
    dataset: fo.Dataset,
    pred_field: str,
    gt_field: str,
    evaluation_params: dict = EVALUATION_PARAMS,
) -> tp.Dict[str, dict]:
    ground_truth_zip_buffer, prediction_zip_buffer, has_transcription, has_confidence = create_tedeval_labels(dataset,
                                                                                                              pred_field,
                                                                                                              gt_field)
    evaluation_params["CONFIDENCES"] = has_confidence
    evaluation_params["TRANSCRIPTION"] = has_transcription
    return evaluate_zip_text_detections(ground_truth_zip_buffer, prediction_zip_buffer, evaluation_params)
