import numpy as np
from dataclasses import dataclass
from typing import *

from scipy.constants import Planck, speed_of_light, elementary_charge
from scipy.integrate import odeint, solve_ivp

import matplotlib.pyplot as plt

from style import set_style

Scaler = float
Integer = int


@dataclass
class Transition:
    kernel: Optional[np.ndarray]
    rate: Scaler
    start: Integer
    end: Integer

@dataclass
class Parameters:
    excitation : Callable[[Scaler], np.ndarray]
    transitions : List[Transition]
    initial_population : np.ndarray
    capacity: np.ndarray
    energy : np.ndarray

@dataclass
class Results:
    populations : List[np.ndarray]
    spectral_fluxes : List[np.ndarray]
    fluxes : List[np.ndarray]
    times : np.ndarray


def run(parameters : Parameters, time_span : np.ndarray) -> Results:

    def advance(time, population):
        fluxes, _ = compute_flux(population, time, parameters)
        return fluxes

    def compute_flux(population, time, parameters):
        fluxes = np.zeros_like(population)
        spectral_flux_total = np.zeros_like(parameters.energy)
        dlambda = parameters.energy[1] - parameters.energy[0]

        for transition in parameters.transitions:
            p_start = population[transition.start]
            p_end = population[transition.end]

            cap = parameters.capacity[transition.end]

            effective_rate = transition.rate * p_start * (cap - p_end)

            if not (transition.kernel is None):
                spectral_rate = parameters.excitation(time) * transition.kernel
                spectral_flux = spectral_rate * effective_rate

                flux = np.trapz(spectral_flux, parameters.energy, dlambda)
            else:
                spectral_flux = np.zeros_like(spectral_flux_total)
                flux = effective_rate

            spectral_flux_total += spectral_flux
            fluxes[transition.start] -= flux
            fluxes[transition.end] += flux

        return fluxes, spectral_flux_total

    def compute_fluxes(populations, times):
        flux_list = []
        spectral_flux_list = []

        for pop, time in zip(populations, times):
            fluxes, spectral_flux_total = compute_flux(pop, time, parameters)

            flux_list.append(fluxes)
            spectral_flux_list.append(spectral_flux_total)

        return np.array(flux_list), np.array(spectral_flux_list)

    sol = solve_ivp(advance, (np.min(time_span), np.max(time_span)),
                     parameters.initial_population,
                    dense_output=True, t_eval=time_span, method="Radau")

    pop = np.transpose(sol.y)
    flux_list, spectral_flux_list = compute_fluxes(pop, sol.t)

    return Results(pop, spectral_flux_list, flux_list, sol.t)

def l_e(l):
    h = Planck
    c = speed_of_light
    q = elementary_charge

    return (h * c / (l * 1e-9)) / q

def gaussian(x, center, sigma):
    return np.exp(-np.power((x - center) / sigma, 2))

def pulse(xs, start, end):
    def F(x):
        if (x > start) and (x < end):
            return 1.0

        return 0.0

    return np.vectorize(F)(xs)


def normalize(x, y):
    N = np.trapz(y, x, x[1] - x[0])
    return y / N

def plot_2d(x, y, z):
    X,Y = np.meshgrid(x, y)
    plt.contourf(X, Y, z, levels = 100)

def three_state_TA():
    t_start = 0.0
    t_end = 40.0

    energy = np.linspace(1, 4, 1000)
    time = np.linspace(t_start, t_end, 1200)

    probe_spectrum = gaussian(energy, l_e(600) - 0.01, 0.1)
    pump_spectrum = gaussian(energy, l_e(400) - 0.01, 0.1)

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
    plt.title("Spectral Overlap")
    plt.plot(energy, kernel, label = "Kernel")
    plt.plot(energy, excitation_spectrum, label = "Excitation")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.savefig("overlap.pdf", dpi=300)


    plt.figure(figsize=figure_size)
    plt.title("Max Absorbance vs. Excitation Envelope")
    plt.plot(results.times, spectral_flux[:, np.argmax(spectral_flux, axis=1)[0]], label="Response")
    plt.plot(results.times, excitation_env(results.times), label="Excitation")
    plt.xlabel("Time (ns)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.savefig("absorbance.pdf", dpi=300)

    plt.figure(figsize=figure_size)
    plot_2d(energy, results.times, spectral_flux)
    plt.title("Total Spectral Flux")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Time (ns)")
    plt.savefig("spectral_flux.png", dpi=300)

    plt.figure(figsize=figure_size)
    plt.title("Populations")
    plt.plot(results.times, results.populations[:, 0], label="Ground State")
    plt.plot(results.times, results.populations[:, 1], label="Excited State")
    plt.legend()
    plt.savefig("populations.pdf", dpi=300)


    plt.show()

    print()




if __name__ == '__main__':
    three_state_TA()
