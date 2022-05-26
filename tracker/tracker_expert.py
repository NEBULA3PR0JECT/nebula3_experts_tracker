from os import getenv
from fastapi import FastAPI

from nebula3_experts.experts.service.base_expert import BaseExpert
from nebula3_experts.experts.app import ExpertApp
from nebula3_experts.experts.common.models import ExpertParam
from tracker.common.models import StepParam
from tracker.common.config import TRACKER_CONF
from nebula3_experts.experts.common.defines import OutputStyle
import tracker.autotracker as at

""" Predict params
@param: predict_every: how many frames to track before accepting detection model detections.
@param: merge_iou_threshold: the IOU score threhold for merging items during tracking.
@param: refresh_on_detect: if True, removes all tracked items that were not found by the detection model.
@param: tracker_type - TRACKER_TYPE_KCF / TRACKER_TYPE_CSRT
@param: batch_size
@param: step - array of steps: [detect,track,depth]
"""


class TrackerExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        self.config = TRACKER_CONF()
        self.model = self.load_model()
        self.tracker_dispatch_dict = {}
        # after init all
        self.set_active()

    def load_model(self):
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

        return at.detection_utils.VideoPredictor(getattr(at.detection_utils, self.args.model),
                                                         confidence_threshold=self.args.confidence)

    def get_name(self):
        return "TrackerExpert"

    def add_expert_apis(self, app: FastAPI):
        @app.post("/detect")
        def post_detect(detect_params: StepParam):
            result = {}
            if detect_params.movie_id is None:
                self.logger.error(f'missing movie_id')
                return { 'error': f'movie frames not found: {detect_params.movie_id}'}
            try:
                movie, num_frames = self.get_movie_and_frames(detect_params.movie_id)
                # now detect
                self.detect(detect_params)
            except Exception as e:
                error_msg = f'exception: {e.message()} on movie: {detect_params.movie_id}'
                self.logger.error(error_msg)
                result['error'] = error_msg
            finally:
              self.tracker_dispatch_dict.pop(detect_params.movie_id)
            return {"result": result }

        @app.post("/track")
        def post_track(track_params: StepParam):
            result = {}
            if track_params.movie_id is None:
                self.logger.error(f'missing movie_id')
                return { 'error': f'movie frames not found: {track_params.movie_id}'}
            try:
                movie, num_frames = self.get_movie_and_frames(track_params.movie_id)
                # now track
                self.track(track_params)
            except Exception as e:
                error_msg = f'exception: {e.message()} on movie: {track_params.movie_id}'
                self.logger.error(error_msg)
                result['error'] = error_msg
            finally:
              self.tracker_dispatch_dict.pop(track_params.movie_id)
            return {"result": result }

        @app.post("/depth")
        def post_depth(track_params: StepParam):
            return {"track_params": track_params }

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        movie = self.movie_db.get_movie(expert_params.movie_id)
        print(f'Predicting movie: {expert_params.movie_id}')
        return { 'result': { 'movie_id' : expert_params.movie_id, 'info': movie , 'extra_params': expert_params.extra_params} }

    def get_movie_and_frames(self, movie_id: str):
        """get movie from db and load movie frames from remote if not exists
        Args:
            movie_id (str): the movie id

        Raises:
            ValueError: _description_

        Returns:
            _type_: movie and number of frames
        """
        movie = self.movie_db.get_movie(movie_id)
        self.tracker_dispatch_dict[movie_id] = {}
        # loading the movie frames
        num_frames = self.movie_s3.downloadDirectoryFroms3(movie_id)
        if num_frames == 0:
            raise ValueError(f'no frames found under the name {movie_id}')
        return (movie, num_frames)

    def detect(self, detect_params: StepParam):
        """detector step

        Args:
            detect_params (StepParam): _description_

        Returns:
            aggs: _description_
        """
        aggs = {}
        self.model.predict_video(detect_params.movie_id,
                                 batch_size = detect_params.batch_size,
                                 pred_every = detect_params.predict_every,
                                 show_pbar = False,
                                 global_aggregator = aggs)
        return aggs
    def detect(self, detect_params: StepParam):
        pass

tracker_expert = TrackerExpert()
expert_app = ExpertApp(expert=tracker_expert)
app = expert_app.get_app()
expert_app.run()