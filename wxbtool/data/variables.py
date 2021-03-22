# -*- coding: utf-8 -*-

vars2d = [
    '2m_temperature',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    'total_cloud_cover', 'total_precipitation',
    'toa_incident_solar_radiation',
    'temperature_850hPa',
]

vars3d = [
    'geopotential', 'temperature',
    'specific_humidity', 'relative_humidity',
    'u_component_of_wind', 'v_component_of_wind',
    'vorticity', 'potential_vorticity',
]

codes = {
    'geopotential': 'z',
    'temperature': 't',
    'temperature_850hPa': 't',
    'specific_humidity': 'q',
    'relative_humidity': 'r',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'vorticity': 'vo',
    'potential_vorticity': 'pv',
    '2m_temperature': 't2m',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    'total_cloud_cover': 'tcc',
    'total_precipitation': 'tp',
    'toa_incident_solar_radiation': 'tisr',
}

code2var = {
    'z': 'geopotential',
    't': 'temperature',
    'q': 'specific_humidity',
    'r': 'relative_humidity',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'vo': 'vorticity',
    'pv': 'potential_vorticity',
    't2m': '2m_temperature',
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'tcc': 'total_cloud_cover',
    'tp': 'total_precipitation',
    'tisr': 'toa_incident_solar_radiation',
}


def split_name(composite):
    if composite == 't2m' or composite == 'u10' or composite == 'v10' or composite == 'tcc' or composite == 'tp' or composite == 'tisr':
        return composite, ''
    else:
        if composite[:2] == 'vo' or composite[:2] == 'pv':
            return composite[:2], composite[2:]
        else:
            return composite[:1], composite[1:]
