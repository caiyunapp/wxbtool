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
