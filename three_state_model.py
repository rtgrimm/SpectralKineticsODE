import numpy as np
import matplotlib.pyplot as plt
from style import set_style
from kinetics import *
from util import *

def three_state():
    t_start = 0.0
    t_end = 40.0

    energy = np.linspace(1, 4, 1000)
    time = np.linspace(t_start, t_end, 1200)

    initial_populations = np.array([
        1.0, 0.0, 0.0, 0.0
    ])

    cap = np.array([
        1.0, 1.0, 1.0, 1.0
    ])

    G = 0
    A = 1
    C = 2
    trap = 3

    k_decay = 1.0
    k_exc = 1000.0
    k_trap = 0.1
    k_an = 10.0

    G_A_kernel = normalize(energy, gaussian(energy, 2.0, 0.05))
    G_C_kernel = normalize(energy, gaussian(energy, 2.6, 0.05))
    A_C_kernel = normalize(energy, gaussian(energy, 2.3, 0.05)) * 100.0


    transitions = [
        Transition(G_A_kernel, k_exc, G, A),
        Transition(G_C_kernel, k_exc, G, C),
        Transition(A_C_kernel, k_exc, A, C),
        Transition(None, k_decay * 1.0, A, G),
        Transition(None, k_decay * 2.0, C, A),
    ]


    probe_spectrum = gaussian(energy, 2.0 - 0.01, 0.1)
    pump_spectrum = gaussian(energy, 2.6 - 0.01, 0.1)

    params = Parameters(lambda t: gaussian(t, 2.0, 0.01) * probe_spectrum * 0.1, transitions, initial_populations, cap, energy)
    result_A_0 = run(params, time)
    A_0 = np.sum(result_A_0.spectral_fluxes, axis=0)

    plt.plot(energy, A_0)
    plt.show()

    spectra = []
    tau_list = np.linspace(2.0, 40.0, 20)

    pump = lambda t: gaussian(t, 2.0, 0.1)

    for tau in tau_list:
        print(f"{(tau / np.max(tau_list)) * 100.0}%")

        probe = lambda t: gaussian(t, tau, 0.01)

        exc_total = lambda t: pump(t) * pump_spectrum + probe(t) * probe_spectrum * 0.1
        params = Parameters(exc_total, transitions, initial_populations, cap, energy)

        results = run(params, time)

        spectra.append(np.sum(results.spectral_fluxes, axis=0) - A_0)

    spectra = np.array(spectra)

    set_style()
    plt.figure(figsize=(16, 9))

    plt.subplot(2, 1, 1)
    plot_2d(tau_list, energy, np.transpose(spectra))
    plt.colorbar()
    plt.xlabel("$\\tau$ (ns)")
    plt.ylabel("Energy (eV)")
    plt.xlim(0.0, np.max(time))

    params = Parameters(lambda t: pump(t) * pump_spectrum, transitions, initial_populations, cap, energy)
    result_simple = run(params, time)

    plt.subplot(2, 1, 2)
    plt.plot(result_simple.times, result_simple.populations[:, 0], label="G")
    plt.plot(result_simple.times, result_simple.populations[:, 1], label="A")
    plt.plot(result_simple.times, result_simple.populations[:, 2], label="C")
    #plt.plot(result_simple.times, result_simple.populations[:, 3], label="Trap")
    plt.legend()
    plt.ylabel("Population")
    plt.xlabel("Time (ns)")
    plt.xlim(0.0, np.max(time))


    plt.tight_layout()
    plt.savefig("TA.png", dpi=300)


    plt.show()

