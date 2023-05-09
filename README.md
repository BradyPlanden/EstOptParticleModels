# Estimate and Optimise Particle Models

## Running with Docker:

First build the PyBaMM docker image via `docker build -t pybamm-base`

Next, create a container from the image via `sudo docker run -it -p 8888:8888 --name pybamm-opt -v /path/to/this/repo:/home/PyBaMM-Opt pybamm-base bash`

Run the container: `docker start -ai PyBaMM-Opt` and navigate to `/home/PyBaMM-Opt/`

Finally, run the benchmark `python experiment/NLOpt-Fitting.py`
