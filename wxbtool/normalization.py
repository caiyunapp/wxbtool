# -*- coding: utf-8 -*-


def norm_gpt(x):
    min, max = -10000, 500000
    return (x - min) / (max - min)


def norm_tmp(x):
    min, max = 173, 373
    return (x - min) / (max - min)


def norm_shm(x):
    x = x * (x > 0)
    min, max = 0, 0.1
    return (x - min) / (max - min)


def norm_rhm(x):
    x = x * (x > 0)
    min, max = 0.0, 200.0
    return (x - min) / (max - min)


def norm_u(x):
    min, max = -250.0, 250.0
    return (x - min) / (max - min)


def norm_v(x):
    min, max = -250.0, 250.0
    return (x - min) / (max - min)


def denorm_gpt(x):
    return 510000 * x - 10000


def denorm_tmp(x):
    return 173 + 200.0 * x


def denorm_shm(x):
    return 0.1 * x


def denorm_rhm(x):
    return 200.0 * x


def denorm_u(x):
    return x * 500 - 250.0


def denorm_v(x):
    return x * 500 - 250.0


normalizors = {
    'geopotential': norm_gpt,
    'temperature': norm_tmp,
    'specific_humidity': norm_shm,
    'relative_humidity': norm_rhm,
    'u_component_of_wind': norm_u,
    'v_component_of_wind': norm_v,
}


denormalizors = {
    'geopotential': denorm_gpt,
    'temperature': denorm_tmp,
    'specific_humidity': denorm_shm,
    'relative_humidity': denorm_rhm,
    'u_component_of_wind': denorm_u,
    'v_component_of_wind': denorm_v,
}
