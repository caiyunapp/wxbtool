# -*- coding: utf-8 -*-


import numpy as np
import cv2

from threading import local
from wxbtool.plot.cmaps import cmaps

data = local()


def imgdata():
    if 'img' in dir(data):
        return data.img
    else:
        data.img = np.zeros([32, 64, 4], dtype=np.uint8)
        return data.img


def colorize(data, out, cmap):
    data = data.reshape(32, 64)
    fliped = (data[::-1, :] * 255).astype(np.uint8)
    return np.take(cmaps[cmap], fliped, axis=0, out=out)


def imsave(fileobj, data):
    is_success, img = cv2.imencode(".png", data)
    buffer = img.tobytes()
    fileobj.write(buffer)


def plot(fileobj, data, cmap='coolwarm'):
    imsave(fileobj, colorize(data, imgdata(), cmap))
