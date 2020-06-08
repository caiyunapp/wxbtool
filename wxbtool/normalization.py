# -*- coding: utf-8 -*-

import torch as th


def norm_gpt(x):
    min, max = -10000, 500000
    return (x - min) / (max - min)


def norm_tmp(x):
    min, max = 173, 373
    return (x - min) / (max - min)


def norm_shm(x):
    x = th.relu(x)
    min, max = 0, 0.1
    return (x - min) / (max - min)


def norm_rhm(x):
    x = th.relu(x)
    min, max = 0.0, 200.0
    return (x - min) / (max - min)


def norm_u(x):
    min, max = -250.0, 250.0
    return (x - min) / (max - min)


def norm_v(x):
    min, max = -250.0, 250.0
    return (x - min) / (max - min)


normalizors = {
    'geopotential': norm_gpt,
    'temperature': norm_tmp,
    'specific_humidity': norm_shm,
    'relative_humidity': norm_rhm,
    'u_component_of_wind': norm_u,
    'v_component_of_wind': norm_v,
}
