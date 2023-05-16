FROM python:3.8.16

WORKDIR /usr/src/app

RUN apt-get update --yes && \
    apt-get install --yes \
    python3-pip \
    python3-dev \
    build-essential \
    ghostscript \
    libgs-dev \
    libmagickwand-dev \
    imagemagick --fix-missing \
    libpng-dev \
    zlib1g-dev \
    libjpeg-dev \
    gfortran \ 
    libopenblas-dev \
    liblapack-dev \
    vim


# Pip install python packages
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Jupyter notebook port
EXPOSE 8888

# Start Jupyter lab on container start
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
