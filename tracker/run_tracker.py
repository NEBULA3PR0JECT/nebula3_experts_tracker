import autotracker as at
import pprint

# create a default detection model according to conda env.
model = at.detection_utils.VideoPredictor()


import os

# configure an experiment
experiment_config  = dict(
    detect_every=10,          # use detection model every `detect_every` frames (starting from 0) 
    merge_iou_threshold=0.5,  # required IOU score 
    tracker_type=at.tracking_utils.TRACKER_TYPE_KCF,  # trecker algorithm (KCF or CSRT)
    refresh_on_detect=False   # if True, any object that isn't found by the model is removed
)

def convert_avi_to_mp4(avi_file_path, output_name):
        os.system("ffmpeg -y -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(
            input=avi_file_path, output=output_name))
        return True

input_cwd = os.getcwd()
# convert_avi_to_mp4(os.path.join(input_cwd,"tracker/movies/actioncliptrain00188.avi"), os.path.join(input_cwd,"tracker/movies/actioncliptrain00188"))

# choose movie. save in appropriate experiment fiile
input_path = os.path.join(input_cwd,"tracker/movies/1024_Identity_Thief_00_01_43_655-00_01_47_807.mp4")
output_path = os.path.splitext(os.path.basename(input_path))[0] + f'__{",".join(f"{k}={v}" for k, v in experiment_config.items())}'

from TrackerAnnotator import TrackerAnnotator

# do tracking with experiment config.
tracking_data = at.tracking_utils.MultiTracker.track_video_objects(input_path,
                                                                   model,
                                                                   **experiment_config)
# annotate video when tracking is complete.
print(f"Saving the annotated video in: {output_path}")
if tracking_data != {}:
    TrackerAnnotator().annotate_video(input_path, tracking_data, output_path)
else:
    print("Invalid video path.")

if '0' in tracking_data:
    pprint.pprint(f"Example of output data (ID: 1): {tracking_data['2']}")


# Template
# {
#     "movie_id": "Movies/114208196",
#     "scene_element": 1,
#     "expert": "tracker",
#     "IDs": {"0":
#                 "class": "person",
#                 "scores": 
#                     'frame_start': 0.99
#                     ...
#                     'frame_end': 0.99
#                 "bboxes": 
#                     'frame_start': 0.99
#                     ...
#                     'frame_end': 0.99
#                 "frame_start": 0
#                 "frame_end": 44
#             ...
#             "10":
#                 "class": "tie",
#                 "scores": 
#                     'frame_start': 0.97
#                     ...
#                     'frame_end': 0.95
#                 "bboxes": 
#                     'frame_start': 0.98
#                     ...
#                     'frame_end': 0.95
#                 "frame_start": 0
#                 "frame_end": 7  
             
#              }

# }