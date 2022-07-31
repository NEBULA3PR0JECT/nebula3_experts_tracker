from .tracker_model import TrackerModel
from .detectron_tracker_model import DetectronTrackerModel
from .yolo_tracker_model import YoloTrackerModel

from typing import List

def get_tracker_model_list() -> List[str]:
    return ["detectron","yolo"]


def get_tracker_model(type: str) -> TrackerModel:
    if type == "detectron":
        return DetectronTrackerModel
    elif type == "yolo":
        return YoloTrackerModel
    else:
        raise ValueError(f"Unsupported model type '{type}'.")


def create_tracker_model(type: str) -> TrackerModel:
    model_class = get_tracker_model(type)
    return model_class()
