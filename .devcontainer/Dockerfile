FROM python:3.10
USER root

RUN apt-get install debian-archive-keyring -y && \
    apt-key update


RUN apt-get update --allow-releaseinfo-change && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip && \
    apt-get install -y sudo wget vim curl gawk make gcc

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN sudo apt-get update && \
    sudo apt-get install iputils-ping

EXPOSE 8501