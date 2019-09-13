# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile-gpu
# FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04
# Google cloud fails with cuda 10
FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         wget \
         curl \
         python \
         python3 \
         python3-dev \
         build-essential \
         python3-pip && \
     rm -rf /var/lib/apt/lists/*


# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

RUN pip3 install setuptools

# RUN cd /usr/local/bin \
#   && ln -s /usr/bin/python3 python \
#   && pip3 install --upgrade pip

WORKDIR /root

# Installs pytorch and torchvision.
RUN pip3 install torch==1.2 torchvision==0.4 google-cloud-storage==1.19.0 

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip3 install cloudml-hypertune  # NOTE: It defaulted to python2

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
# RUN mkdir /root/trainer
# COPY trainer/mnist.py /root/trainer/mnist.py
COPY luminoth /luminoth

# Sets up the entry point to invoke the trainer.
WORKDIR /luminoth

# Download models now so they wont have to be downloaded each time we run google cloud
RUN python3 -c "import torchvision.models as models; models.resnet50(pretrained=True)"

# RUN apt-get install -y --no-install-recommends python3-dev
# RUN pip3 install numpy cython  # Try deleting
RUN pip3 install git+git://github.com/tryolabs/cocoapi.git#subdirectory=PythonAPI
ENTRYPOINT python3 train.py --data-path gs://luminothv2/ws_cropped_coco/ --epochs 25 --lr-steps 22 24 --lr 0.001 -b 3 -j 0
