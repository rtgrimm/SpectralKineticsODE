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
    order: Scaler = 1.0

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
    spectral_flux_map: Dict[Tuple[Integer, Integer], np.ndarray]
    fluxes : List[np.ndarray]
    times : np.ndarray


def run(parameters : Parameters, time_span : np.ndarray) -> Results:

    def advance(time, population):
        fluxes, _, _ = compute_flux(population, time, parameters)
        return fluxes

    def compute_flux(population, time, parameters):
        fluxes = np.zeros_like(population)
        spectral_flux_total = np.zeros_like(parameters.energy)
        dlambda = parameters.energy[1] - parameters.energy[0]

        spectral_flux_map = {}

        for transition in parameters.transitions:
            p_start = population[transition.start]
            p_end = population[transition.end]

            cap = parameters.capacity[transition.end]

            effective_rate = transition.rate * np.power(p_start, transition.order) * (cap - p_end)

            if not (transition.kernel is None):
                spectral_rate = parameters.excitation(time) * transition.kernel
                spectral_flux = spectral_rate * effective_rate

                flux = np.trapz(spectral_flux, parameters.energy, dlambda)
            else:
                spectral_flux = np.zeros_like(spectral_flux_total)
                flux = effective_rate

            spectral_flux_map[(transition.start, transition.end)] = spectral_flux

            spectral_flux_total += spectral_flux
            fluxes[transition.start] -= flux
            fluxes[transition.end] += flux

        return fluxes, spectral_flux_total, spectral_flux_map

    def compute_fluxes(populations, times):
        flux_list = []
        spectral_flux_list = []
        spectral_flux_maps = []

        spectral_flux_map_all = {}

        for pop, time in zip(populations, times):
            fluxes, spectral_flux_total, spectral_flux_map = compute_flux(pop, time, parameters)

            spectral_flux_maps.append(spectral_flux_map)
            flux_list.append(fluxes)
            spectral_flux_list.append(spectral_flux_total)

        for map in spectral_flux_maps:
            for key, value in map.items():
                if key in spectral_flux_map_all:
                    spectral_flux_map_all[key].append(value)
                else:
                    spectral_flux_map_all[key] = [value]

        for key, value in spectral_flux_map_all.items():
            spectral_flux_map_all[key] = np.array(value)


        return np.array(flux_list), np.array(spectral_flux_list), spectral_flux_map_all

    sol = solve_ivp(advance, (np.min(time_span), np.max(time_span)),
                     parameters.initial_population,
                    dense_output=True, t_eval=time_span, method="Radau")

    pop = np.transpose(sol.y)
    flux_list, spectral_flux_list, spectral_flux_map = compute_fluxes(pop, sol.t)

    return Results(pop, spectral_flux_list, spectral_flux_map, flux_list, sol.t)