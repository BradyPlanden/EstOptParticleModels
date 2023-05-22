# Build Instructions

1. Change directory to /docker/botorch/
2. Build image via `sudo docker build -t pybamm-jupyter-botorch .` 
    1. Ensure the base image is locally available via `docker run jupyter/minimal-notebook:latest`
3. Build container from image via `docker run -it --rm -p 8888:8888 --user root -e CHOWN_HOME=yes -e CHOWN_HOME_OPTS='-R' -v "${PWD}":/home/jovyan/work pybamm-jupyter-botorch`