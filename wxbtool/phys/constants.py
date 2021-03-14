# -*- coding: utf-8 -*-

import numpy as np


g = 9.80665

Omega = 2 * np.pi / (24 * 3600 * 0.99726966323716)

gamma = 6.49 / 1000
gammad = 9.80 / 1000
cv = 718.0
cp = 1005.0
Rs = 287
miu = 1.72e-1
M = 0.02896968 # molar mass of dry air, 0.0289644 kg/mol

T0 = 298.16
R0 = 8.314462618 # Universal gas constant
rao0 = 1.2252256827617731

niu = 1.85e-5 # kinematic viscosity of air in 25 degree C
kappa = 1.9e-5 # thermal diffusivity of air in 300K

SunConst = 1366

StefanBoltzmann = 0.0000000567

WaterHeatCapacity = 4185.5
RockHeatCapacity = 840

WaterDensity = 1000
IceDensity = 916.7
RockDensity = 2650

LatenHeatIce = 334
LatenHeatWater = 2264.705
