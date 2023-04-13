import os 
import pybamm
import time
import numpy as np
import matplotlib.pyplot as plt
from common.optim import *
from models.LNMOParams import *
plt.rcParams["figure.figsize"] = (10,5.65)

# Model definition + loading Chen as default
model = pybamm.lithium_ion.SPM()
params = pybamm.ParameterValues("Chen2020")

# Updating params for LNMO 
DirNeg = f'data/lnmo-exercise/OCV_Graphite.parquet'#"/Users/bradyplanden/Documents/Git/2023-oslo-workshop/lnmo-exercise/OCV_Graphite.parquet"
DirPos = f'data/lnmo-exercise/LNMO_theta_OCV_singleSweep.parquet' #"/Users/bradyplanden/Documents/Git/2023-oslo-workshop/lnmo-exercise/LNMO_theta_OCV_singleSweep.parquet"
LNMOParams(DirPos, DirNeg, params)

# Creating ground-truth data for estimation
params.update(
        {"Electrode height [m]": 0.04727}
        )
experiment =  pybamm.Experiment(
    [
        ("Discharge at 1C for 5 minutes (10 second period)",
        "Rest for 2 minutes (10 second period)",
        "Charge at 0.5C for 2.5 minutes (10 second period)",
        "Rest for 2 minutes (10 second period)"),
    ] * 10
)
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
sol = sim.solve()["Terminal voltage [V]"].data
sol += np.random.normal(0,0.005,len(sol))

# Optimisation Function with L2Norm 
def forward(x, grad):
    output = 2.5 * np.ones(len(sol)) 
    params.update({"Electrode height [m]": x[0]})
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
    new_sol = sim.solve()["Terminal voltage [V]"].data
    output[:len(new_sol)] = new_sol
    return sum((output - sol) ** 2)

# Optimise
t0 = time.time()
x, minf, count = optimiser([0.065], forward, 1e-5,[0.03],[0.1])
t1 = time.time()
total = t1-t0
print("Optimisation Time", total)
print("Number of Evalutions", count)
print("optimum at ", x)
print("minimum value = ", minf)

# Run estimated parameters
params.update(
        {"Electrode height [m]": x[0]}
        )
optsol = sim.solve()["Terminal voltage [V]"].data

# Plotting
plt.figure(1)
plt.plot(sol, label='Groundtruth')
plt.plot(optsol, label='Estimated')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.savefig('sol.png')

plt.figure(2)
plt.plot(sim.solution['Current [A]'].data)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.savefig('sol1.png')
