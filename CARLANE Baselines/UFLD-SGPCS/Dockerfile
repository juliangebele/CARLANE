FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
RUN apt-get update
WORKDIR /tmp
COPY . /tmp
RUN pip install -r requirements.txt
RUN python /tmp/setup.py develop