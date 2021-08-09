import numpy as np
from dataclasses import dataclass
from typing import *

from scipy.constants import Planck, speed_of_light, elementary_charge
from scipy.integrate import odeint, solve_ivp

import matplotlib.pyplot as plt

from kinetics import Scaler, Parameters, run
from style import set_style


@dataclass
class TAParams:
    pump_spectrum : np.ndarray
    probe_spectrum : np.ndarray
    time : np.ndarray
    tau_list : np.ndarray
    probe_time_env : Callable[[Scaler], Callable[[Scaler], Scaler]]
    pump_time_env: Callable[[Scaler], Callable[[Scaler], Scaler]]
    make_params : Any
    probe_scale = 0.1

@dataclass
class TAResults:
    spectra : np.ndarray

def run_TA(TA_params : TAParams) -> TAResults:
    params = TA_params.make_params(
        lambda t: TA_params.probe_time_env(
            np.min(TA_params.tau_list))(t) * TA_params.probe_spectrum * 0.1)

    pump_spectrum = TA_params.pump_spectrum
    probe_spectrum = TA_params.probe_spectrum

    result_A_0 = run(params, TA_params.time)
    A_0 = np.sum(result_A_0.spectral_fluxes, axis=0)



    spectra = []

    pump = TA_params.pump_time_env(np.min(TA_params.tau_list))

    for tau in TA_params.tau_list:
        print(f"{(tau / np.max(TA_params.tau_list)) * 100.0}%")

        probe = TA_params.pump_time_env(tau)

        exc_total = lambda t: pump(t) * pump_spectrum + probe(t) * probe_spectrum * 0.1
        params = TA_params.make_params(exc_total)

        results = run(params, TA_params.time)

        spectra.append(np.sum(results.spectral_fluxes, axis=0) - A_0 * 0)

    spectra = np.array(spectra)

    return TAResults(spectra)