from os import getenv

"""backend & models
VideoPredictor init params:
- tracker backend: detectron or tflow
- pretrained_model_cfg - default: CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x
- confidence_threshold - default 0.5

"""

class TRACKER_CONF:
    def __init__(self) -> None:

        # self.BACKEND_TFLOW = getenv('TFLOW_BACKEND','CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2')
        # self.BACKEND_DETECTRON = getenv('DETECTRON_BACKEND','CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x')
        self.TRACKER_EXPERT = getenv('TRACKER_EXPERT', 'detectron')
        self.BACKEND_TFLOW = getenv('TFLOW_BACKEND','CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2')
        self.BACKEND_DETECTRON = getenv('DETECTRON_BACKEND','CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x')
        self.BACKEND = getenv('BACKEND',None)
        self.BACKENDS = { 'tflow': self.BACKEND_TFLOW, 'detectron': self.BACKEND_DETECTRON }
        self.CONFIDENCE_THRESHOLD = eval(getenv('CONFIDENCE_THRESHOLD','0.5'))

    def get_tracker_expert(self):
        return (self.TRACKER_EXPERT)
    def get_backend_tflow(self):
        return (self.BACKEND_TFLOW)
    def get_backend_detectron(self):
        return (self.BACKEND_DETECTRON)
    def get_backend(self):
        return (self.BACKEND)
    # def get_backends(self):
    #     return self.BACKENDS
    def get_confidence(self):
        return (self.CONFIDENCE_THRESHOLD)
