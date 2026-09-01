from astropy.io import fits
import numpy as np
from pathlib import Path

def bin_func(file : Path, bin_factor : int) -> tuple:
    data,header = fits.getdata(file,header=True)
    m,n = data.shape
    mcut = -(data.shape[0]%bin_factor) if data.shape[0]%bin_factor > 0 else m
    ncut = -(data.shape[1]%bin_factor) if data.shape[1]%bin_factor > 0 else n
    bin_funcs = {"mean" : np.mean,"median" : np.median,"sum" : np.sum}
    data = data[0:mcut,0:ncut]
    reshaped_data = data.reshape(m // bin_factor, bin_factor, 
                                 n // bin_factor, bin_factor)
    # wcs correction
    header["CD1_1"] *= bin_factor
    header["CD1_2"] *= bin_factor
    header["CD2_1"] *= bin_factor
    header["CD2_2"] *= bin_factor
    header["CRPIX1"] //= bin_factor
    header["CRPIX2"] //= bin_factor

    return np.sum(np.sum(reshaped_data,axis=1),axis=2), header