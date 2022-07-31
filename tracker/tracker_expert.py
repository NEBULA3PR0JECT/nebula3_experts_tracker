import os
import sys
import json
from fastapi import FastAPI
from PIL import Image
import cv2
sys.path.append("/notebooks/nebula3_experts")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline")
sys.path.append("/notebooks/nebula3_experts/nebula3_pipeline/nebula3_database")

from nebula3_experts.experts.common.constants import OUTPUT_DB, TYPE_IMAGE, TYPE_MOVIE
from nebula3_experts.experts.service.base_expert import BaseExpert, DEFAULT_FILE_PATH
from nebula3_experts.experts.app import ExpertApp
from nebula3_experts.experts.common.models import ExpertParam, TokenRecord, ImageRecord
from tracker.common.models import StepParam, ImageStepParam
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
            movie_id = expert_params.id if  expert_params.id else ''
            return { 'error': f'error {error} for movie: {movie_id}'}
        print(f'Predicting movie: {expert_params.id} with action: {step_param.action}')
        if step_param.action == ACTION_TRACK:
            result, error = self.handle_action_on_movie(step_param, False, self.track, self.transform_tracking_result)
        if step_param.action == ACTION_DETECT:
            result, error = self.handle_action_on_movie(step_param, False, self.detect, self.transform_movie_frame_result)
        if step_param.action == ACTION_DEPTH:
            pass
        if not error and expert_params.output == OUTPUT_DB:
            result, error = self.save_to_db(expert_params.id, result)
        return { 'result': result, 'error': error }

    def predict_image(self, expert_params: ExpertParam):
        """ predicding a single image """
        result = None
        params, error = self.parse_image_detection_params(expert_params)
        try:
            self.add_task(params.image_id, params.__dict__)
            # downloading the image
            img_fetched = self.download_image_file(params.image_url)
            if img_fetched:
                # load image
                img = cv2.imread(DEFAULT_FILE_PATH)
                # running detection
                prediction = self.model.predict_single_frame(img)
                if prediction:
                    result = self.transform_detection_result([prediction], TYPE_IMAGE, params.image_id)
                else:
                    error = f'failed to run detection on image id: {params.image_id}, url: {params.image_url}'
                    self.logger.error(error)
            else:
                error = f'failed to download image id: {params.image_id}, url: {params.image_url}'
                self.logger.error(error)
        except Exception as e:
            result = False
            error = f'exception: {e} on image: {params.image_id}'
            self.logger.error(error)
        finally:
            self.remove_task(params.image_id)

        return { 'result': result, 'error': error }

    def parse_image_detection_params(self, expert_params: ExpertParam):
        error = None
        if (expert_params.id is None):
            error = 'no image id'
            return None, error
        if (expert_params.img_url is None):
            error = 'no image url'
            return None, error

        img_step_param = ImageStepParam(image_id=expert_params.id,
                                        image_url=expert_params.img_url,
                                        output=expert_params.output)
        if expert_params.extra_params:
            if 'merge_iou_threshold' in expert_params.extra_params:
                img_step_param.merge_iou_threshold = expert_params.extra_params['merge_iou_threshold']
            if 'refresh_on_detect' in expert_params.extra_params:
                img_step_param.refresh_on_detect = expert_params.extra_params['refresh_on_detect']
            if 'tracker_type' in expert_params.extra_params:
                img_step_param.tracker_type = expert_params.extra_params['tracker_type']
        return img_step_param, error

    def parse_tracker_params(self, expert_params: ExpertParam):
        error = None
        if (expert_params.id is None):
            error = 'no movie id'
            return None, error
        if (expert_params.extra_params is None):
            error = 'no extra_params id'
            return None, error
        step_param = StepParam(movie_id=expert_params.id,
                               output=expert_params.output,
                               scene_element=expert_params.scene_element)
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
        if 'mdf' in expert_params.extra_params:
            step_param.mdf = expert_params.extra_params['mdf']

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

    def detect(self, detect_params: StepParam):
        """detector step
        resolve the frame to detect and save it,
        prepare the ds for the frame and predict it
        Args:
            detect_params (StepParam): _description_

        Returns:
            aggs: _description_
        """
        scene_element = detect_params.scene_element if detect_params.scene_element else 0
        movie = self.movie_db.get_movie(detect_params.movie_id)
        if scene_element > len(movie['scene_elements']):
            return None, f'scene_element: {scene_element} is bigger than movie scene elements'
        frame_number = movie['mdfs'][scene_element][detect_params.mdf]
        frames = self.divide_movie_into_frames([frame_number])
        # img = Image.open(frames[0])
        img = cv2.imread(frames[0])
        prediction = self.model.predict_single_frame(img)
        return [prediction]
        # return self.model.predict_video(DEFAULT_FILE_PATH, # detect_params.movie_id,
        #                          batch_size = detect_params.batch_size,
        #                          pred_every = detect_params.detect_every,
        #                          show_pbar = False)

    def transform_movie_frame_result(self, detection_result, detect_params: StepParam):
        return self.transform_detection_result(detection_result, TYPE_MOVIE, detect_params.movie_id)

    def transform_detection_result(self, detection_result, detect_type, id):
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
                if detect_type is TYPE_MOVIE:
                    tr = TokenRecord(id,
                                    0, 0, self.get_name(),
                                    detections[cls],
                                    cls,
                                    {'class': 'Object'})
                else: # type image
                    tr = ImageRecord(id,
                                    self.get_name(),
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