from fastapi import FastAPI

from nebula3_experts.service.base_expert import BaseExpert
from nebula3_experts.app import ExpertApp
from nebula3_experts.common.models import ExpertParam

""" Predict params
@param: detect_every: how many frames to track before accepting detection model detections.
@param: merge_iou_threshold: the IOU score threhold for merging items during tracking.
@param: refresh_on_detect: if True, removes all tracked items that were not found by the detection model.
@param: tracker_type - TRACKER_TYPE_KCF / TRACKER_TYPE_CSRT
@param: batch_size
"""

"""backend & models
VideoPredictor init params:
- tracker backend: detectron or tflow
- pretrained_model_cfg - default: CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x
- confidence_threshold - default 0.5

CFG_DEFAULT = {
    at.BACKEND_TFLOW: 'CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2',
    at.BACKEND_DETECTRON: 'CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x'
}
"""

class TrackerExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        # after init all
        self.set_active()
    def get_name(self):
        return "TrackerExpert"

    def add_expert_apis(self, app: FastAPI):
        pass
        # @app.get("/my-expert")
        # def get_my_expert(q: Optional[str] = None):
        #     return {"expert": "my-expert" }

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        movie = self.movie_db.get_movie(expert_params.movie_id)
        print(f'Predicting movie: {expert_params.movie_id}')
        return { 'result': { 'movie_id' : expert_params.movie_id, 'info': movie , 'extra_params': expert_params.extra_params} }


tracker_expert = TrackerExpert()
expert_app = ExpertApp(expert=tracker_expert)
app = expert_app.get_app()
expert_app.run()