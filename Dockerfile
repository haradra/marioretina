FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /tmp

# install packages by updating base env in conda
COPY docker-env.yml /tmp
COPY . /tmp
RUN conda env update -f docker-env.yml

# solution to cv2 shared lib issue
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'gcc'\ 
    'libxext6'  -y

CMD ["/bin/bash"]