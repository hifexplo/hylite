from gfit import mgauss
import numpy as np
import hylite
import skimage.data as skdata
from skimage.transform import rotate, swirl, rescale


def randomSpectra(wav: np.ndarray,
                  f: list = [2245., 2340.],
                  d: list = [0.2, 0.3],
                  w: list = [50, 100],
                  a: list = 0.9,
                  nrand: list = 15,
                  noise: list = 0.05,
                  slope: list = 0.2):
    """
    Generate a random spectra.
    :param wav: The wavelengths to generate a spectra across.
    :param f: Fixed absorption feature positions.
    :param d: Fixed absorption feature depths.
    :param w: Fixed absorption feature widths.
    :param a: Overall material brightness (from 0 to 1). Default is 0.9.
    :param nrand: Number of random features to add.
    :param noise: Depth of random features and additional white noise.
    :param slope: Slope of overall spectra.
    :return: A synthetic spectra
    """
    # generate initial (pure) spectra
    R = (1 - slope * (wav - wav[0]) / (wav[-1] - wav[0]))  # slope
    R2 = np.zeros_like(wav)
    mgauss(wav, R2, d, f, w)
    R = R - R2

    if nrand > 0:
        mgauss(wav, R2, [np.random.uniform(0., noise) for i in range(nrand)],
               [np.random.uniform(wav[0], wav[-1]) for i in range(nrand)],
               [np.random.uniform(10 * (wav[1] - wav[0]), (wav[-1] - wav[0])) for i in range(nrand)])
        R = a * (R - R2)
    return np.clip(R + np.random.rand(len(wav)) * noise, 0., 1.)


def genImage(wav: np.ndarray = np.linspace(2100., 2400., 200),
             A: dict = dict(f=[2200., 2340.], a=1.0, nrand=0),
             B: dict = dict(f=[2160., 2210.], a=0.5, nrand=0),
             C: dict = dict(f=[2250., 2350.], a=0.2, nrand=0),
             flip: bool = False, seed: int = 42, noise: float = 0.05):
    """
    Use the Bricks() and Stones() images in skimage.data to create a synthetic hyperspectral image. Useful
    for testing or demonstration purposes.
    :param wav: Numpy array containing wavelengths of the desired HSI image. Default is np.linspace(2100., 2400., 200).
    :param A: A dictionary containing arguments passed to randomSpectra(...) for phase A. Default is {f=[2200., 2340.], a=1.0}.
    :param B: A dictionary containing arguments passed to randomSpectra(...) for phase B. Default is {f=[2160., 2200.], a=0.5}.
    :param C: A dictionary containing arguments passed to randomSpectra(...) for phase C. Default is {f=[2250., 2350.], a=0.2}.
    :param flip: True if the order of latent variables should be reversed. Useful for generating different styles of image.
    :param seed: An integer seed for numpy.random(...). Default is 42.
    :param noise: Amount of white noise to add. Default is 0.05.
    :return: The synthetic hyperspectral image
    """

    np.random.seed(42)

    # Define two latent variables
    lA = (255 - np.clip(2 * skdata.brick(), 0, 255)) / 255.
    lA = swirl(rotate(lA, 15, mode='symmetric'), strength=2, radius=1024)
    lB = swirl(rescale(np.tile(skdata.grass() / 255., (2, 2)), 0.5), strength=2, radius=1024)

    # Combine into three closed abundances (sum to 1)
    ab = np.dstack([lA, lB, 1.0 - (lA + lB)])
    ab[ab < 0] = 0.01
    ab = ab ** 2  # enhance contrast a little
    ab /= np.sum(ab, axis=-1)[..., None]

    # Sample endmember spectra
    eA = randomSpectra(wav, **A)  # Phase A
    eB = randomSpectra(wav, **B)  # Phase B
    eC = randomSpectra(wav, **C)  # Phase C

    # mix to generate fake HSI data
    hsi = (ab.reshape(-1, 3) @ np.array([eA, eB, eC])).reshape((lA.shape) + (len(wav),))
    hsi += np.random.rand(*hsi.shape) * noise
    # put image and abundances in a HyImage object and return
    hsi = hylite.HyImage(hsi)
    hsi.set_wavelengths(wav)
    A = hylite.HyImage(ab)
    A.set_band_names(['mineral_A', 'mineral_B', 'mineral_C'])
    return hsi, A

