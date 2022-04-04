# follow instructions from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
# tutorial: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb

load_base_env
echo "==== Activating conda environment ====="
conda create -n $TFLOW_ENV_NAME python=3.8 -y
conda activate $TFLOW_ENV_NAM

# install tensorflow
echo "===== installing tensorflow ====="
pip install tensorflow
pip install tf_slim

# install coco tools
echo "===== installing COCO tools ====="
pip install pycocotools

# get models repo
echo "===== cloning models repository ====="
git clone --depth 1 https://github.com/tensorflow/models  # todo fork

# get protoc 3.3 to fix issue stated here: https://github.com/tensorflow/models/issues/1834
mkdir protoc_3.3
pushd protoc_3.3
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
chmod 775 protoc-3.3.0-linux-x86_64.zip
unzip -o protoc-3.3.0-linux-x86_64.zip
popd

# install object_detection API
echo "===== installing object detection API ====="
pushd models/research/
# using previously downloaded protoc 3.3
../../protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

# downgrade to tensorflow 2.4 (CUDA issue)
# pip install tensorflow==2.4
pip3 install tensorflow==2.1.0

# test setup
echo "===== testing setup ====="
python object_detection/builders/model_builder_tf2_test.py
popd


function load_base_env {
    # load base directly
    source "$(dirname $(dirname $(which conda)))/bin/activate"

    # check for errors
    if [[ "$?" != "0" ]] || [[ $CONDA_DEFAULT_ENV != "base" ]]; then
        echo "ERROR: loading conda. please run with conda \"base\" env"
        exit 1
    fi
}