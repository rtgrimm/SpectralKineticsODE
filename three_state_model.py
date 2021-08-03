import numpy as np
from dataclasses import dataclass
from typing import *

from scipy.constants import Planck, speed_of_light, elementary_charge
from scipy.integrate import odeint, solve_ivp

import matplotlib.pyplot as plt

from style import set_style
from kinetics import *
from util import *

def three_state():
    t_start = 0.0
    t_end = 40.0

    energy = np.linspace(1, 4, 1000)
    time = np.linspace(t_start, t_end, 1200)



    k_decay = 0.01
    k_exc = 1000

    gs_kernel = normalize(energy, gaussian(energy, l_e(600), 0.05))
    es_kernel = normalize(energy, gaussian(energy, l_e(400), 0.05))

    initial_populations = np.array([
        1.0, 0.0, 0.0
    ])

    cap = np.array([
        10.0, 1.0, 1.0
    ])

    gs = 0
    es_1 = 1
    es_2 = 2

    transitions = [
        Transition(gs_kernel, k_exc, gs, es_1),
        Transition(es_kernel, k_exc, gs, es_2),
        Transition(None, k_decay, es_1, gs),
        Transition(None, k_decay * 1e2, es_2, es_1),
    ]

    spectra = []
    tau_list = np.linspace(2.0, 40.0, 20)

    probe_spectrum = gaussian(energy, l_e(600) - 0.01, 0.1)
    pump_spectrum = gaussian(energy, l_e(400) - 0.01, 0.1)

    pump = lambda t: gaussian(t, 2.0, 0.1)

    for tau in tau_list:
        print(tau)


        probe = lambda t: gaussian(t, tau, 0.01)

        exc_total = lambda t: pump(t) * pump_spectrum + probe(t) * probe_spectrum * 0.1

        #plt.plot(energy, gs_kernel)
        #plt.plot(energy, es_kernel)
        #plt.show()

        params = Parameters(exc_total, transitions, initial_populations, cap, energy)

        results = run(params, time)

        spectra.append(np.sum(results.spectral_fluxes, axis=0))
        #plt.plot(energy, spectra[0])
        #plt.show()

    spectra = np.array(spectra)

    set_style()
    plt.figure(figsize=(16, 9))

    plt.subplot(2, 1, 1)
    plot_2d(tau_list, energy, np.transpose(spectra))
    plt.xlabel("$\\tau$ (ns)")
    plt.ylabel("Energy (eV)")
    plt.xlim(0.0, np.max(time))

    params = Parameters(lambda t: pump(t) * pump_spectrum, transitions, initial_populations, cap, energy)
    result_simple = run(params, time)

    plt.subplot(2, 1, 2)
    plt.plot(result_simple.times, result_simple.populations[:, 0], label="GS")
    plt.plot(result_simple.times, result_simple.populations[:, 1], label="ES 1")
    plt.plot(result_simple.times, result_simple.populations[:, 2], label="ES 2")
    plt.legend()
    plt.ylabel("Population")
    plt.xlabel("Time (ns)")
    plt.xlim(0.0, np.max(time))


    plt.show()
