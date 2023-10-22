EVALUATION_PARAMS = {
    "RECALL_IOU_THRESHOLD": 0.4,
    "PRECISION_IOU_THRESHOLD": 0.4,
    "EV_PARAM_IND_CENTER_DIFF_THR": 1,
    "GT_SAMPLE_NAME_2_ID": ".*([0-9]+).*",
    "DET_SAMPLE_NAME_2_ID": ".*([0-9]+).*",
    "GT_LTRB": False,  # LTRB: 2 points (left, top, right, bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
    "GT_CRLF": False,  # Lines are delimited by Windows CRLF format
    "DET_LTRB": False,  # LTRB: 2 points (left, top, right, bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
    "DET_CRLF": False,  # Lines are delimited by Windows CRLF format
    "CONFIDENCES": False,  # Detections must include confidence value. AP will be calculated.
    "TRANSCRIPTION": False,  # Does prediction have transcription or not
    "PER_SAMPLE_RESULTS": True,  # Generate per sample results and produce data for visualization
    "F_BETA": 1,
}
