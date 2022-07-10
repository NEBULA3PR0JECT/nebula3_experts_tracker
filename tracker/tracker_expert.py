import os
import sys
import json
from fastapi import FastAPI

sys.path.append("/notebooks/nebula3_experts")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline/nebula3_database")

from nebula3_experts.experts.common.constants import OUTPUT_DB
from nebula3_experts.experts.service.base_expert import BaseExpert, DEFAULT_FILE_PATH
from nebula3_experts.experts.app import ExpertApp
from nebula3_experts.experts.common.models import ExpertParam, TokenRecord
from tracker.common.models import StepParam
from tracker.common.config import TRACKER_CONF
from nebula3_experts.experts.common.defines import OutputStyle

# sys.path.insert(0,"/notebooks/tracker/common/../..")
# sys.path.insert(1,"/notebooks/tracker")
# sys.path.remove(".")
# remove for microservice, enable for vscode container
# sys.path.remove("/notebooks")

sys.path.append("/notebooks/tracker/autotracker")
sys.path.append("/notebooks/tracker/autotracker/tracking/../../..")

import tracker.autotracker as at

ACTION_DETECT = 'detect'
ACTION_TRACK = 'track'
ACTION_DEPTH = 'depth'
ACTION_ALL = 'all' # track+depth

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
        step_param, error = self.parse_tracker_params(expert_params)
        if error:
            movie_id = expert_params.movie_id if  expert_params.movie_id else ''
            return { 'error': f'error {error} for movie: {movie_id}'}
        print(f'Predicting movie: {expert_params.movie_id} with action: {step_param.action}')
        if step_param.action == ACTION_TRACK:
            result, error = self.handle_action_on_movie(step_param, False, self.track, self.transform_tracking_result)
        if step_param.action == ACTION_DETECT:
            result, error = self.handle_action_on_movie(step_param, False, self.detect, self.transform_detection_result)
        if step_param.action == ACTION_DEPTH:
            pass
        if not error and expert_params.output == OUTPUT_DB:
            result, error = self.save_to_db(expert_params.movie_id, result)
        return { 'result': result, 'error': error }

    def parse_tracker_params(self, expert_params: ExpertParam):
        error = None
        if (expert_params.movie_id is None):
            error = 'no movie id'
            return None, error
        if (expert_params.extra_params is None):
            error = 'no extra_params id'
            return None, error
        step_param = StepParam(movie_id=expert_params.movie_id, output=expert_params.output)
        if expert_params.extra_params['action']:
            step_param.action = expert_params.extra_params['action']
        if expert_params.extra_params['detect_every']:
            step_param.detect_every = expert_params.extra_params['detect_every']
        if 'merge_iou_threshold' in expert_params.extra_params:
            step_param.merge_iou_threshold = expert_params.extra_params['merge_iou_threshold']
        if 'refresh_on_detect' in expert_params.extra_params:
            step_param.refresh_on_detect = expert_params.extra_params['refresh_on_detect']
        if 'tracker_type' in expert_params.extra_params:
            step_param.tracker_type = expert_params.extra_params['tracker_type']
        if 'batch_size' in expert_params.extra_params:
            step_param.batch_size = expert_params.extra_params['batch_size']

        return step_param, error

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
            self.add_task(params.movie_id, params.__dict__)
            if not movie_fetched:
                movie_fetched = self.download_video_file(params.movie_id)
            if movie_fetched:
                # now calling action function
                action_result = action_func(params)
                # now transforming results data
                result = transform_func(action_result, params)
            else:
                error_msg = f'no frames for movie: {params.movie_id}'
                self.logger.warning(error_msg)
        except Exception as e:
            error_msg = f'exception: {e} on movie: {params.movie_id}'
            self.logger.error(error_msg)
        finally:
            self.remove_task(params.movie_id)
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
        return self.model.predict_video(DEFAULT_FILE_PATH, # detect_params.movie_id,
                                 batch_size = detect_params.batch_size,
                                 pred_every = detect_params.detect_every,
                                 show_pbar = False)

    def transform_detection_result(self, detection_result, detect_params: StepParam):
        """transform detection result to the token db format

        Args:
            detection_result (_type_): _description_
            output (_type_): json/db - transforming for json output or for db
        """
        # print(detection_result)
        detections = {}
        result = list()
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
                tr = TokenRecord(detect_params.movie_id,
                                0, 0, self.get_name(),
                                detections[cls],
                                cls,
                                {'class': 'Object'})
                result.append(tr)
        return result

    def track(self, track_params: StepParam):
        track_data = at.tracking_utils.MultiTracker.track_video_objects(
                video_path=DEFAULT_FILE_PATH, #  track_params.movie_id,
                detection_model=self.model,
                detect_every=track_params.detect_every,
                merge_iou_threshold=track_params.merge_iou_threshold,
                tracker_type=track_params.tracker_type,
                refresh_on_detect=track_params.refresh_on_detect,
                show_pbar=False,
                logger=self.logger
            )
        return track_data

    def transform_tracking_result(self, tracking_result, track_params: StepParam):
        # print(tracking_result)
        result = list()
        for oid, data in tracking_result.items():
            label = data['class'] + str(oid)
            print(data['boxes'])
            print(data['scores'])
            print(label)
            bbox = dict()
            for index, box in data['boxes'].items():
                bbox[index] = { 'score': data['scores'][index], 'bbox': box}
            tr = TokenRecord(track_params.movie_id,
                             0, 0, self.get_name(),
                             bbox,
                             label,
                             {'class': 'Object'})
            result.append(tr)
        return result

tracker_expert = TrackerExpert()
expert_app = ExpertApp(expert=tracker_expert)
app = expert_app.get_app()
expert_app.run()