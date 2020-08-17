# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np


# Land-sea mask
def load_lsm(resolution, root):
    data_path = '%s/constants/constants_%s.nc' % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose('lat', 'lon')
    dt = np.array(ds['lsm'].data, dtype=np.float32)
    return dt


# Soil type
def load_slt(resolution, root):
    data_path = '%s/constants/constants_%s.nc' % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose('lat', 'lon')
    dt = np.array(ds['slt'].data, dtype=np.float32)
    return dt


# Orography
def load_orography(resolution, root):
    data_path = '%s/constants/constants_%s.nc' % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose('lat', 'lon')
    dt = np.array(ds['orography'].data, dtype=np.float32)
    return dt


# Latitude
def load_lat2d(resolution, root):
    data_path = '%s/constants/constants_%s.nc' % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose('lat', 'lon')
    dt = np.array(ds['lat2d'].data, dtype=np.float64)
    return dt


# Longitude
def load_lon2d(resolution, root):
    data_path = '%s/constants/constants_%s.nc' % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose('lat', 'lon')
    dt = np.array(ds['lon2d'].data, dtype=np.float64)
    return dt


# Area weight
def load_area_weight(resolution, root):
    data_path = '%s/constants/constants_%s.nc' % (root, resolution)
    ds = xr.open_dataset(data_path)
    ds = ds.transpose('lat', 'lon')
    lat = np.array(ds['lat2d'].data, dtype=np.float64)
    dt = np.cos(lat * np.pi / 180)
    dt = dt / dt.mean()
    return dt



