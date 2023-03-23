import os 
import pybamm
import numpy as np
import pandas as pd
from common.optim import *

def LNMOParams(DirPos, DirNeg, params):
    ocp_neg = pd.read_parquet(DirPos) #"/Users/bradyplanden/Documents/Git/2023-oslo-workshop/lnmo-exercise/OCV_Graphite.parquet"
    ocp_pos = pd.read_parquet(DirNeg) #"/Users/bradyplanden/Documents/Git/2023-oslo-workshop/lnmo-exercise/LNMO_theta_OCV_singleSweep.parquet"
    ocp_pos.OCV = ocp_pos.values.OCV[::-1]

    # OCP Functions
    def graphite_OCP(theta):
        return pybamm.Interpolant(np.array(ocp_neg["theta"]),np.array(ocp_neg["OCV"]),theta)

    def lnmo_OCP():
        return pybamm.Interpolant(np.array(ocp_pos["theta"]),np.array(ocp_pos["OCV"]),theta)

    # Updating params
    params.update(
        {"Positive electrode OCP [V]":lnmo_OCP,
        "Negative electrode OCP [V]": graphite_OCP,
        "Upper voltage cut-off [V]": 5.0}
    )

def PybammForward(model, experiment, params):
    sim = pybamm.Simulation(model,experiment=experiment, parameter_values=params)
    return sim.solve()["Terminal voltage [V]"].data