import autotracker as at
from TrackerAnnotator import TrackerAnnotator
import os

# create a default detection model according to conda env.
# model = at.detection_utils.VideoPredictor()

# # configure an experiment
# experiment_config  = dict(
#     detect_every=10,          # use detection model every `detect_every` frames (starting from 0) 
#     merge_iou_threshold=0.5,  # required IOU score 
#     tracker_type=at.tracking_utils.TRACKER_TYPE_KCF,  # tracker algorithm (KCF or CSRT)
#     refresh_on_detect=False   # if True, any object that isn't found by the model is removed
# )

# choose movie. save in appropriate experiment fiile
cwd = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(cwd, 'movies/actioncliptrain00188.avi')
# output_path = os.path.splitext(os.path.basename(input_path))[0] + f'__{",".join(f"{k}={v}" for k, v in experiment_config.items())}'

# # do tracking with experiment config.
# tracking_data = at.tracking_utils.MultiTracker.track_video_objects(input_path,
#                                                                    model,
#                                                                    **experiment_config)
# # annotate video when tracking is complete.
# TrackerAnnotator().annotate_video(input_path, tracking_data, output_path)

print("Num GPUs Available: ", at.gpu_count())
output_path = os.path.splitext(os.path.basename(input_path))[0] + '__OUT'
# choose model cfg that is compatible with the current model backend
# The model backend is chosen according to your curent active conda environment:
#   - engines: detectron2
#   - tflow:   tensorflow
# Here are some examples of configurations for available backends.
if at.active_detection_backend() == at.BACKEND_DETECTRON:  # detectron models
    cfg = at.detection_utils.CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x  # best detection
    # cfg = at.detection_utils.CFG_COCO_PANOPTIC_FPN_R101_LR3x  # best panoptic
    # cfg = at.detection_utils.CFG_LVIS_SEGMENTATION_MRCNN_X101_32x8d_FPN_LR1x  # best LVIS
    # cfg = at.detection_utils.CFG_CITISCAPE_SEGMENTATION_MRCNN_R50_FPN  # best cityscape
else:  # tensorflow models
    cfg = at.detection_utils.CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2  # OID (600 classes)
    # cfg = at.detection_utils.CFG_COCO_DETECTION_SSD_MOBILENET_V1  # COCO (92 classes)

# create model with config
predictor = at.detection_utils.VideoPredictor(cfg)

preds = predictor.predict_video(
    path_to_video=input_path,
    # batch_size=4,           # you can control the batch size. This may cause CUDA OOM error
    # pred_every=1,           # run on every `pred_every` frames, starting from frame 0 (first)
    # show_pbar=True,         # set to `False` to hide the progress bar
    # global_aggregator=None  # put predictions directly in some external aggregator (maybe Queue)
)

# the returned value is a list of predictoins, one for each processed frame.
# a prediction is a dictionary in format:
# {
#   'detection_boxes': a list of lists. each list is a box for a single object in format xywh,
#   'detection_scores': a list of floating points. this is the confidence for each object,
#   'detection_classes': a list of string classes, e.g. "person", "car", "mango", ...
# }
# Print first prediction
print(preds)

