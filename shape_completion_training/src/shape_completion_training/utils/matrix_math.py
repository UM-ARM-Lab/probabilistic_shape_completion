"""
I hate that I have to write these...
scipy has these implemented, but because I am using python 2 with tensorflow I am using scipy 1.2.2, which
does not implement ".as_matrix"

ugh...
"""
import numpy as np


def rotx(rad):
    s = np.sin(rad)
    c = np.cos(rad)

    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])


def roty(rad):
    s = np.sin(rad)
    c = np.cos(rad)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])


def rotz(rad):
    s = np.sin(rad)
    c = np.cos(rad)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def rotzyx(xr, yr, zr, degrees=False):
    if degrees:
        xr = xr * 3.1415 / 180
        yr = yr * 3.1415 / 180
        zr = zr * 3.1415 / 180
    mx = rotx(xr)
    my = roty(yr)
    mz = rotz(zr)
    return np.matmul(mx, np.matmul(my, mz))


def rotxyz(xr, yr, zr, degrees=False):
    if degrees:
        xr = xr * 3.1415 / 180
        yr = yr * 3.1415 / 180
        zr = zr * 3.1415 / 180
    mx = rotx(xr)
    my = roty(yr)
    mz = rotz(zr)
    return np.matmul(mz, np.matmul(my, mx))
