# Download the NVIDIA container image for Tensorflow for python3. Note that
# although we use the tensorflow image, this also contains the TensorRT library
# (and the TF-TRT code that interfaces the two software packages).
#
# NVIDIA Docker container support matrix:
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/
#
# NOTE: The TensorRT containers available from NVIDIA only contain TensorRT and
# not TensorFlow. These are not suitable for training, but are suitable for
# inference (and much smaller in size overall). For reference, the corresponding
# image/tag for an inference container compatible with "tensorflow:xx.xx-yy-py3"
# would be "tensorrt:xx.xx-py3".

# Define NVIDIA Docker container and Framework
FROM nvcr.io/nvidia/tensorflow:20.06-tf1-py3
ARG FRAMEWORK=tensorflow1

# Define username for AirPack Docker
ARG USERNAME=deepwave
ARG UID=1000

# Don't allow apt-get to prompt user when building a container,
# there is no controlling tty
ARG DEBIAN_FRONTEND=noninteractive

# Install python TensorRT python interface, update the repo, add some Python
# packages, upgrade pip, and add the new user
RUN apt-get update -qq && \
    apt-get install -qq -y python3-pip ipython3 pciutils sudo && \
    pip3 -q install --upgrade pip && \
    useradd -u $UID -ms /bin/bash $USERNAME

# Install TensorRT python interface
RUN /opt/tensorrt/python/python_setup.sh

# Use USERNAME user by default from here forward
USER $USERNAME
WORKDIR /home/$USERNAME

ADD requirements.txt /home/$USERNAME/requirements.txt
RUN pip install --only-binary "onnx" -r /home/$USERNAME/requirements.txt && rm /home/$USERNAME/requirements.txt

# Expose a port for netron to serve model visualizations on
EXPOSE 8080

RUN printf "\n\n\AirPack successfully installed.\n"
