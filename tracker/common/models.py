from pydantic import BaseModel
import nebula3_experts.experts.common.constants as constants

class StepParam(BaseModel):
    movie_id: str
    scene_element: int = 0
    mdf: int = 0
    action: str = 'track'
    detect_every: int = 10
    merge_iou_threshold: float = 0.95
    refresh_on_detect: bool = True
    tracker_type: str =  'KCF'
    batch_size: int = 8
    output: str = constants.OUTPUT_JSON

class ImageStepParam(BaseModel):
    image_id: str
    image_url: str
    merge_iou_threshold: float = 0.95
    refresh_on_detect: bool = True
    tracker_type: str =  'KCF'
    output: str = constants.OUTPUT_JSON
