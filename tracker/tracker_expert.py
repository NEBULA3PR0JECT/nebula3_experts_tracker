import os
import sys
import json
from fastapi import FastAPI
from nebula3_experts.experts.service.base_expert import BaseExpert
from nebula3_experts.experts.app import ExpertApp
from nebula3_experts.experts.common.models import ExpertParam
from tracker.common.models import StepParam
from tracker.common.config import TRACKER_CONF
from nebula3_experts.experts.common.defines import OutputStyle

sys.path.insert(0,"/notebooks/tracker/common/../..")
sys.path.insert(1,"/notebooks/tracker")
sys.path.remove(".")
sys.path.remove("/notebooks")
# sys.path.append("/notebooks/tracker/autotracker")
# sys.path.append("/notebooks/tracker/autotracker/tracking/../../..")

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

    def get_name(self):
        return "TrackerExpert"

    def add_expert_apis(self, app: FastAPI):
        @app.post("/detect")
        def post_detect(detect_params: StepParam):
            if detect_params.movie_id is None:
                self.logger.error(f'missing movie_id')
                return { 'error': f'movie frames not found: {detect_params.movie_id}'}
            result, error = self.handle_action_on_movie(detect_params, False, self.detect, self.transform_detection_result)
            return { 'result': result, 'error': error }

        @app.post("/track")
        def post_track(track_params: StepParam):
            if track_params.movie_id is None:
                self.logger.error(f'missing movie_id')
                return { 'error': f'movie frames not found: {track_params.movie_id}'}
            result, error = self.handle_action_on_movie(track_params, False, self.track, self.transform_tracking_result)
            return { 'result': result, 'error': error }

        @app.post("/depth")
        def post_depth(track_params: StepParam):
            return {"track_params": track_params }

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        movie = self.movie_db.get_movie(expert_params.movie_id)
        print(f'Predicting movie: {expert_params.movie_id}')
        return { 'result': { 'movie_id' : expert_params.movie_id, 'info': movie , 'extra_params': expert_params.extra_params} }

    def handle_action_on_movie(self,
                               params: StepParam,
                               movie_fetched: bool,
                               action_func,
                               transform_func):
        """handing detection/tracking/depth on movie

        Args:
            params (StepParam): _description_
            movie_fetched (_type_): indicated if a movie and it's frames already fetched
            since this method can be called for each type: detection/tracking/etc'

        Returns:
            result or error
        """
        error_msg = None
        result = None
        if params.movie_id is None:
            self.logger.error(f'missing movie_id')
            return { 'error': f'movie frames not found: {params.movie_id}'}
        try:
            if not movie_fetched:
                movie, num_frames = self.get_movie_and_frames(params.movie_id)
            if movie and num_frames:
                # now calling action function
                action_result = action_func(params)
                # now transforming results data
                result = transform_func(action_result, params.output)
            else:
                error_msg = f'no frames for movie: {params.movie_id}'
                self.logger.warning(error_msg)
        except Exception as e:
            error_msg = f'exception: {e} on movie: {params.movie_id}'
            self.logger.error(error_msg)
        finally:
            self.tracker_dispatch_dict.pop(params.movie_id)
        return result, error_msg

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
        return (movie, num_frames)

    def detect(self, detect_params: StepParam):
        """detector step

        Args:
            detect_params (StepParam): _description_

        Returns:
            aggs: _description_
        """
        return self.model.predict_video(detect_params.movie_id,
                                 batch_size = detect_params.batch_size,
                                 pred_every = detect_params.detect_every,
                                 show_pbar = False)

    def transform_detection_result(self, detection_result, output):
        """transform detection result to the token db format

        Args:
            detection_result (_type_): _description_
            output (_type_): json/db - transforming for json output or for db
        """
        # print(detection_result)
        detections = {}
        for detection in detection_result:
            detection_boxes = detection['detection_boxes']
            detection_scores = detection['detection_scores']
            detection_classes = detection['detection_classes']
            for idx in range(len(detection_classes)):
                cls = detection_classes[idx]
                bbox = detection_boxes[idx]
                score = detection_scores[idx]
                element = {'bbox': bbox.tolist(), 'score': float(score.flat[0]) }
                if cls in detections:
                    detections[cls].append(element)
                else:
                    detections[cls] = [element]
        return detections

    def track(self, track_params: StepParam):
        track_data = at.tracking_utils.MultiTracker.track_video_objects(
                video_path=track_params.movie_id,
                detection_model=self.model,
                detect_every=track_params.detect_every,
                merge_iou_threshold=track_params.merge_iou_threshold,
                tracker_type=track_params.tracker_type,
                refresh_on_detect=track_params.refresh_on_detect,
                show_pbar=False,
                logger=self.logger
            )
        return track_data

    def transform_tracking_result(self, tracking_result, output):
        # print(tracking_result)
        result = {}
        for oid, data in tracking_result.items():
            print(data['boxes'])
            print(data['scores'])
            print(data['class'] + str(oid))
            result[data['class'] + str(oid)] = {'boxes': data['boxes'], 'scores': data['scores'] }
        return result

tracker_expert = TrackerExpert()
expert_app = ExpertApp(expert=tracker_expert)
app = expert_app.get_app()
expert_app.run()