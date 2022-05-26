from pydantic import BaseModel
from nebula3_experts.experts.common.defines import OutputStyle

class StepParam(BaseModel):
    movie_id: str
    detect_every: int = 10
    merge_iou_threshold: float = 0.95
    refresh_on_detect: bool = True
    tracker_type: str =  'KCF'
    batch_size: int = 8
    output: str = OutputStyle.JSON


