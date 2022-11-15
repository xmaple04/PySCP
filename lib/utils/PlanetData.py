"""
地球运行相关的参数
"""
import numpy as np


class Earth:
    mu = 3.986004418e14
    radius = 6378137
    g0 = mu / radius ** 2
    v0 = np.sqrt(mu / radius)  # 7905.365719014348m/s
    period = 2 * np.pi * np.sqrt(radius ** 3 / mu)  # 5069.343798881842s
    rho0 = 1.225  # kg/m3
    H0 = 7200  # 大气参考高度,m
    P0 = 101325  # N/m2, Pa
    beta = 1 / H0


class Moon:
    mu = 4.900105726362566e12
    radius = 1737400  # average radius
    g0 = mu / radius ** 2
    v0 = np.sqrt(mu / radius)  # 1579.4m/s
    period = 2 * np.pi * np.sqrt(radius ** 3 / mu)  # 6500.2s


class Mars:
    """
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
    """
    mu = 4.282837e13
    radius = 3396200
    ellipticity = 0.00589
    g0 = 3.71
    J2 = 1.96045e-3


if __name__ == '__main__':
    moon = Moon()
