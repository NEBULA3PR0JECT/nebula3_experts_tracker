from .tracker_model import TrackerModel
from tracker.common.config import TRACKER_CONF

class YoloTrackerModel(TrackerModel):

    def __init__(self):
        super().__init__()
        self.config = TRACKER_CONF()

    def forward(self, image, metadata=None):
        pass
