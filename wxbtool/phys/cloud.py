import numpy as np
import torch as th


from wxbtool.phys.constants import WaterDensity, IceDensity


def water_mass_by_density(d):
    l = d / 1000
    return np.pi * l * l * l / 6 * WaterDensity


def ice_mass_by_density(d):
    l = d / 1000
    return np.pi * l * l * l / 6 * IceDensity


def water_reflectivity_by_density(d):
    return d * d * d * d * d * d


def ice_reflectivity_by_density(d):
    return 1.5 * d * d * d * d * d * d


def marshall_palmer(n0, d):
    return n0 * th.exp(-4.1 * d)
