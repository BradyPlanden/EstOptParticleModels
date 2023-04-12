import os 
import pybamm
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
        {"Electrode height [m]": 0.04727, 
         "Negative particle radius [m]": 0.4e-6, 
         "Positive particle radius [m]":0.6e-5}
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
    params.update({"Electrode height [m]": x[0], "Negative particle radius [m]": x[1], "Positive particle radius [m]": x[2]})
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
    new_sol = sim.solve()["Terminal voltage [V]"].data
    output[:len(new_sol)] = new_sol
    return sum((output - sol) ** 2)


# Optimise
x, minf = optimiser([0.065, 0.3e-6, 0.3e-5], forward, 1e-5,[0.03, 0.1e-6, 0.1e-5],[0.1, 0.8e-6, 0.8e-5])
print("optimum at ", x)
print("minimum value = ", minf)

# Run estimated parameters
params.update(
        {"Electrode height [m]": x[0], 
         "Negative particle radius [m]": x[1], 
         "Positive particle radius [m]":x[2]}
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
