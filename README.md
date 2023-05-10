# Estimate and Optimise Particle Models

## Running with Docker:

First build the PyBaMM docker image by running `docker build -t pybamm-base` within the /docker directory

Next, create a container from the image via `sudo docker run -it -p 8888:8888 --name pybamm-opt -v /path/to/this/repo:/home/PyBaMM-Opt pybamm-base bash`

Run the container: `docker start -ai PyBaMM-Opt` and navigate to `/home/PyBaMM-Opt/`

Run `pip install -e .`

Finally, run the benchmark `python experiment/NLOpt-Fitting.py`
