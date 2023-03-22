import pybamm
import numpy as np
from common.optim import *

# Forward Model Initialisation
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
        {"Electrode height [m]": 0.04727, "Negative particle radius [m]": 0.4e-6, "Positive particle radius [m]":0.6e-5}
        )
experiment =  pybamm.Experiment(
    [
        ("Discharge at 1C for 5 minutes (10 second period)",
        "Rest for 2 minutes",
        "Charge at 0.5C for 2.5 minutes (10 second period)",
        "Rest for 2 minutes"),
    ] * 10
)
sim = pybamm.Simulation(model,experiment=experiment, parameter_values=parameter_values)
sol = sim.solve()["Terminal voltage [V]"].data

# Gaussian noise
s = np.random.normal(0,0.005,len(sol))
sol += s

def forward(x, grad):
    output = 2.5 * np.ones(len(sol)) 
    parameter_values.update({"Electrode height [m]": x[0], "Negative particle radius [m]": x[1], "Positive particle radius [m]": x[2]})
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
    new_sol = sim.solve()["Terminal voltage [V]"].data
    output[:len(new_sol)] = new_sol
    return sum((output - sol) ** 2)

# Optimise
x, minf = optimiser([0.065, 0.3e-6, 0.3e-5], forward, 1e-5,[0.03, 0.1e-6, 0.1e-5],[0.1, 0.8e-6, 0.8e-5])
print("optimum at ", x)
print("minimum value = ", minf)
