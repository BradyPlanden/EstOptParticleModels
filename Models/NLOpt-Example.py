import nlopt
import pybamm
import numpy as np

# Forward Model Initialisation
model = pybamm.lithium_ion.SPM()
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
        {"Electrode height [m]": 0.04727, "Negative particle radius [m]": 0.4e-6, "Positive particle radius [m]":0.6e-5}
        )
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve([0, 1800], inputs={"I": 2})["Terminal voltage [V]"].data

#Gaussian noise
s = np.random.normal(0,0.005,100)
sol += s

def forward(x, grad):
    output = 2.5 * np.ones(100) 
    parameter_values.update({"Electrode height [m]": x[0], "Negative particle radius [m]": x[1], "Positive particle radius [m]": x[2]})
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    new_sol = sim.solve([0, 1800])["Terminal voltage [V]"].data
    output[:len(new_sol)] = new_sol
    return sum((output - sol) ** 2)

opt = nlopt.opt(nlopt.LN_BOBYQA, 3)
opt.set_min_objective(forward)
opt.set_xtol_rel(1e-5)
opt.set_lower_bounds([0.03, 0.1e-6, 0.1e-5])
opt.set_upper_bounds([0.1, 0.8e-6, 0.8e-5])
x = opt.optimize([0.065, 0.3e-6, 0.3e-5])
minf = opt.last_optimum_value()
print("optimum at ", x)
print("minimum value = ", minf)
print("result code = ", opt.last_optimize_result())
