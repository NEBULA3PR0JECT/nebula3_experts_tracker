# nebula3_experts_tracker
Nebula Tracker expert repository
1. Docker - directory with docker files (Dockerfile)
2. tracker - code for tracker expert
3. nebula3_database - submodule with database integration and API

# How to run from DOCKER
1. `git clone https://github.com/dsivov/nebula3_experts_tracker.git`
2. `nvidia-docker run -it --name detectron2 detectron2:v0`
3. `python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input.jpg --output outputs/ --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl`

The output should be: `input.jpg: detected 15 instances.`
# How to run without docker
1. `git clone https://github.com/dsivov/nebula3_experts_tracker.git`
2. `chmox a+x /tracker/setup_envs.sh`
3. `./setup_envs.sh`
4. `conda activate detectron`
5. Check that everything works by running `python3 tracker/my_test.py` (works on a movie)

Note 1: `my_test.py` works on a movie. and `demo/demo.py` works on an image.

Note 2: If you need to install docker or nvidia-docker, please refer to this link: https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/

# How to create submodule: (Already done in this repo, please skip this)

1. Then get submodule:
`git submodule add https://github.com/NEBULA3PR0JECT/nebula3_database.git`
`git submodule init `

2. Change to the submodule directory:
`cd nebula3_database`

3. Checkout desired branch:
`git checkout main`

4. Update:
git pull

5. Get back to your project root:
`cd ..`

6. Now the submodules are in the state you want, so"
`git commit -am "Pulled down update to submodule_dir"`

7. Add Microservice:
Predict on single movie:
POST http://localhost:8000/predict
{

    "movie_id": "Movies/222509721",
    "local": false,
    "output": "json",
    "other_params": {
        "mdfs_only": true/false,
        "with_tracking": true/false,
        "with_depth": true/false
    }
}