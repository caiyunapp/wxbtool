# -*- coding: utf-8 -*-


import numpy as np
import cv2

from threading import local
from wxbtool.plot.cmaps import cmaps, var2cmap


data = local()


def imgdata():
    if 'img' in dir(data):
        return data.img
    data.img = np.zeros([32, 64, 4], dtype=np.uint8)
    return data.img


def colorize(data, out, cmap):
    data = data.reshape(32, 64)
    data = (data - data.min()) / (data.max() - data.min())
    data = (data * (data >= 0) * (data < 1) + (data >= 1)) * 255
    fliped = (data[::-1, :]).astype(np.uint8)
    return np.take(cmaps[cmap], fliped, axis=0, out=out)


def imsave(fileobj, data):
    is_success, img = cv2.imencode(".png", data)
    buffer = img.tobytes()
    fileobj.write(buffer)


def plot(var, fileobj, data):
    imsave(fileobj, colorize(data, imgdata(), var2cmap[var]))
