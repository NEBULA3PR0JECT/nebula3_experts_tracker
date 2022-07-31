from .tracker_model import TrackerModel
from tracker.common.config import TRACKER_CONF

import tracker.autotracker as at

class DetectronTrackerModel(TrackerModel):

    def __init__(self):
        super().__init__()
        self.config = TRACKER_CONF()
        self.model = self.load_model()

    def load_model(self):
        self.confidence = self.config.CONFIDENCE_THRESHOLD
        self.model = None
        self.backend = self.config.get_backend()
        if self.backend:
            self.model = self.config.BACKENDS[self.backend]

        if self.backend is None and self.model is None:
           self.backend = at.active_detection_backend()
           self.model = self.config.BACKENDS[self.backend]

        # no config chosen. use default config for given backend
        elif self.model is None:
            at.set_active_backend(self.backend)
            self.model = self.config.BACKENDS[self.backend]

        # no backend chosen. find backend that has given config
        elif self.backend is None:
            possible_backends = at.backends_with_config(self.model)
            if not possible_backends:
                raise ValueError(f'Model configuration {self.model} not found in any backend available in this evironment')
            self.backend = possible_backends[0]
            at.set_active_backend(self.args.backend)

        # chosen backend and config. check compatibility
        else:
            at.set_active_backend(self.args.backend)
            if not hasattr(at.detection_utils, self.args.model):
                raise ValueError(f'Given model backend and config are incopatible in the current environment: {self.backend} - {self.model}')

        return at.detection_utils.VideoPredictor(getattr(at.detection_utils, self.model),
                                                         confidence_threshold=self.confidence)

    def forward(self, image, metadata=None):
        return self.model.predict_single_frame(image)

    def save(self, label):
        pass
