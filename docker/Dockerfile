FROM python:3.8.16

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN pip install --upgrade pip
RUN apt-get install -y ghostscript libgs-dev
RUN apt-get install -y libmagickwand-dev imagemagick --fix-missing
RUN apt-get install -y libpng-dev zlib1g-dev libjpeg-dev
RUN apt-get install -y gfortran libopenblas-dev liblapack-dev
RUN apt-get install -y vim

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt