import numpy as np
from dataclasses import dataclass
from typing import *

from scipy.constants import Planck, speed_of_light, elementary_charge
from scipy.integrate import odeint, solve_ivp

import matplotlib.pyplot as plt

from style import set_style
from kinetics import *
from util import *



def two_state_model():
    abs_main = l_e(500)

    t_start = 0.0
    t_end = 20.0

    energy = np.linspace(1, 3, 1000)
    time = np.linspace(t_start, t_end, 1000)

    kernel = normalize(energy, gaussian(energy, abs_main, 0.1))

    excitation_spectrum = gaussian(energy, abs_main - 0.01, 0.5)

    excitation_env = lambda t: gaussian(t, 5, 0.5)*5
    excitation = lambda t: excitation_env(t) * excitation_spectrum

    k_decay = 1
    k_exc = 1000

    initial_populations = np.array([
        1.0, 0.0
    ])

    cap = np.array([
        1.0, 0.8
    ])

    transitions = [
        Transition(kernel, k_exc, 0, 1),
        Transition(None, k_decay, 1, 0)
    ]

    params = Parameters(excitation, transitions, initial_populations, cap, energy)

    results = run(params, time)

    spectral_flux = np.array(results.spectral_fluxes)

    set_style()

    figure_size = (10, 10)

    plt.figure(figsize=figure_size)
    #plt.title("Spectral Overlap")
    plt.plot(energy, kernel, label = "Kernel")
    plt.plot(energy, excitation_spectrum, label = "Excitation")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity")
    plt.legend(frameon=False)
    plt.savefig("overlap.pdf", dpi=300)


    plt.figure(figsize=figure_size)
    #plt.title("Max Absorbance vs. Excitation Envelope")
    plt.plot(results.times, spectral_flux[:, np.argmax(spectral_flux, axis=1)[0]], label="Response")
    plt.plot(results.times, excitation_env(results.times), label="Excitation")
    plt.xlabel("Time (ns)")
    plt.ylabel("Intensity")
    plt.legend(frameon=False)
    plt.savefig("absorbance.pdf", dpi=300)

    plt.figure(figsize=figure_size)
    plot_2d(energy, results.times, spectral_flux)
   # plt.title("Total Spectral Flux")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Time (ns)")
    plt.savefig("spectral_flux.png", dpi=300)

    plt.figure(figsize=figure_size)
    #plt.title("Populations")
    plt.plot(results.times, results.populations[:, 0], label="Ground State")
    plt.plot(results.times, results.populations[:, 1], label="Excited State")
    plt.legend(frameon=False)
    plt.savefig("populations.pdf", dpi=300)


    plt.show()
