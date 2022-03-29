FROM nvidia/cuda:10.1-cudnn7-devel
# To use this Dockerfile:
# 1. `nvidia-docker build -t detectron2:v0 .`
# 2. `nvidia-docker run -it --name detectron2 detectron2:v0`


ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	libpng-dev libjpeg-dev ca-certificates ffmpeg libsm6 libxext6 \
	python3-dev build-essential pkg-config git curl wget automake libtool && \
  rm -rf /var/lib/apt/lists/*


# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "source activate base" > ~/.bashrc

SHELL ["/bin/bash", "-c"]
ARG CONDA_ENV=detectron2
RUN conda --version \ 
    && conda create --name ${CONDA_ENV} python=3.7
RUN conda init
RUN source activate ${CONDA_ENV}

RUN conda update -n base -c defaults conda

# RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install scipy==1.6.0
RUN pip3 install opencv-python
RUN pip3 install tqdm
RUN pip3 install numpy

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch torchvision cython \
	'git+https://github.com/facebookresearch/fvcore'
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 /detectron2_repo
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
# RUN pip install -e /detectron2_repo
RUN cd detectron2_repo
RUN pip install -e /detectron2_repo

RUN mkdir detectron2_repo/outputs

# RUN conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /detectron2_repo

RUN wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# run it:
# python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input.jpg --output outputs/ --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
