import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from style import set_style, pal
from kinetics import *
from util import *

def state_tracking_example():
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
    A_C_kernel = normalize(energy, gaussian(energy, 2.3, 0.05))


    transitions = [
        Transition(G_A_kernel, k_exc, G, A),
        Transition(G_C_kernel, k_exc, G, C),
        Transition(A_C_kernel, k_exc, A, C),
        Transition(None, k_decay * 1.0, A, G),
        Transition(None, k_decay * 2.0, C, A),
    ]


    probe_spectrum = gaussian(energy, 2.6, 0.1)

    params = Parameters(lambda t: gaussian(t, 5.0, 2.0) * probe_spectrum, transitions, initial_populations, cap, energy)
    result_A_0 = run(params, time)

    colors = pal("bright", 6)

    handels =[]

    for id, spectrum in result_A_0.spectral_flux_map.items():
        X, Y = np.meshgrid(energy, result_A_0.times)

        color = next(colors)

        red_patch = Patch(color=color, label=f"${id[0]} \\rightarrow {id[1]}$")
        handels.append(red_patch)
        plt.contour(X, Y, spectrum, levels=10, colors=[color], label="Test")


    plt.xlabel("Energy (eV)")
    plt.ylabel("Time (ns)")
    plt.legend(handles=handels)
    plt.show()


