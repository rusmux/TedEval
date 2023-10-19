import io
import zipfile

import fiftyone as fo

from tedeval.evaluate_zip import evaluate_text_detections_zip


def polyline_to_points(
    polyline: fo.Polyline,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    if len(polyline["points"][0]) == 4:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = polyline["points"][0]
    else:
        detection = polyline.to_detection()
        x_min, y_min, width, height = detection["bounding_box"]
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = (
            (x_min, y_min),
            (x_min + width, y_min),
            (x_min + width, y_min + height),
            (x_min, y_min + height),
        )

    x1, x2, x3, x4 = map(lambda x: int(round(x * image_width)), [x1, x2, x3, x4])
    y1, y2, y3, y4 = map(lambda y: int(round(y * image_height)), [y1, y2, y3, y4])

    return x1, y1, x2, y2, x3, y3, x4, y4


def create_tedeval_labels(
    dataset: fo.Dataset,
    pred_field: str,
    gt_field: str,
) -> tuple[io.BytesIO, io.BytesIO]:
    ground_truth_zip_buffer = io.BytesIO()
    prediction_zip_buffer = io.BytesIO()

    for idx, sample in enumerate(dataset.iter_samples(progress=True)):
        image_height = sample["metadata"]["height"]
        image_width = sample["metadata"]["width"]

        ground_truth_field = sample[gt_field]
        if isinstance(ground_truth_field, fo.Detections):
            ground_truth_field = ground_truth_field.to_polylines()

        with zipfile.ZipFile(ground_truth_zip_buffer, "a") as ground_truth_zip:
            ground_truth_label = ""
            for polyline in ground_truth_field["polylines"]:
                x1, y1, x2, y2, x3, y3, x4, y4 = polyline_to_points(polyline, image_width, image_height)
                ground_truth_label += f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{polyline['label']}\n"
            ground_truth_zip.writestr(f"{idx}.txt", ground_truth_label)

        prediction_field = sample[pred_field]
        if isinstance(prediction_field, fo.Detections):
            prediction_field = prediction_field.to_polylines()

        with zipfile.ZipFile(prediction_zip_buffer, "a") as prediction_zip:
            prediction_label = ""
            for polyline in prediction_field["polylines"]:
                x1, y1, x2, y2, x3, y3, x4, y4 = polyline_to_points(polyline, image_width, image_height)
                prediction_label += f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4}\n"
            prediction_zip.writestr(f"{idx}.txt", prediction_label)

    return ground_truth_zip_buffer, prediction_zip_buffer


def evaluate_text_detections_fiftyone(
    dataset: fo.Dataset,
    pred_field: str,
    gt_field: str,
) -> None:
    ground_truth_zip_buffer, prediction_zip_buffer = create_tedeval_labels(dataset, pred_field, gt_field)

    tedeval_results = evaluate_text_detections_zip(ground_truth_zip_buffer, prediction_zip_buffer, {})

    print(tedeval_results)
