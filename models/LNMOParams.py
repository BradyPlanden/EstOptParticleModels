import os 
import pybamm
import numpy as np
import pandas as pd
from common.optim import *

def LNMOParams(DirPos, DirNeg, params):
    ocp_neg = pd.read_parquet(DirNeg, engine="pyarrow") #"/Users/bradyplanden/Documents/Git/2023-oslo-workshop/lnmo-exercise/OCV_Graphite.parquet"
    ocp_pos = pd.read_parquet(DirPos, engine="pyarrow") #"/Users/bradyplanden/Documents/Git/2023-oslo-workshop/lnmo-exercise/LNMO_theta_OCV_singleSweep.parquet"
    ocp_pos.OCV = ocp_pos.OCV.values[::-1]

    # OCP Functions
    def graphite_OCP(theta):
        return pybamm.Interpolant(np.array(ocp_neg.theta),np.array(ocp_neg.OCV),theta)

    def lnmo_OCP(theta):
        return pybamm.Interpolant(np.array(ocp_pos.theta),np.array(ocp_pos.OCV),theta, extrapolate=False)

    # Updating params
    params.update(
        {"Positive electrode OCP [V]":lnmo_OCP,
        "Negative electrode OCP [V]": graphite_OCP,
        "Upper voltage cut-off [V]": 5.0,
        "Maximum concentration in negative electrode [mol.m-3]": 31370
        }
    )

def PybammForward(model, experiment, params):
    sim = pybamm.Simulation(model,experiment=experiment, parameter_values=params)
    return sim.solve()["Terminal voltage [V]"].data
