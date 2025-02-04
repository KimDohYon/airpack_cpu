# Download the NVIDIA container image for PyTorch for python3.
#
# NVIDIA Docker container support matrix:
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/

# Define NVIDIA Docker container and Framework
FROM python:3.8-slim-buster
ARG FRAMEWORK=pytorch

# Define username for AirPack Docker
ARG USERNAME=deepwave
ARG UID=1000

# Don't allow apt-get to prompt user when building a container,
# there is no controlling tty
ARG DEBIAN_FRONTEND=noninteractive

# Install  python, update the repo, add some Python
# packages, upgrade pip, and add the new user
RUN apt-get update -qq && \
    apt-get install -qq -y python3-pip ipython3 pciutils sudo && \
    pip3 -q install --upgrade pip && \
    useradd -u $UID -ms /bin/bash $USERNAME

# Use USERNAME user by default from here forward
USER $USERNAME
WORKDIR /home/$USERNAME

ADD requirements.txt /home/$USERNAME/requirements.txt
RUN pip install -r /home/$USERNAME/requirements.txt && rm /home/$USERNAME/requirements.txt

# Expose a port for netron to serve model visualizations on
EXPOSE 8080

RUN printf "\n\n\
*******************************************************************************************\n\
AirPack successfully installed. Make sure to mount the AirPack dir using the -v flag \
when you start the docker. \nTo start the AirPack docker run the following command:\n\n\
docker run -it -v <path_to_airpack>:/home/$USERNAME/AirPack --gpus all <container_name>\n\n\
*******************************************************************************************\n\
"
