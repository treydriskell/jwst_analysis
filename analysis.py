"""
Analysis module for processing JWST simulated data from Galacticus.

This module provides functions to:
- Load and process Galacticus HDF5 output files
- Calculate galaxy-halo connection statistics
- Compute UV luminosity functions
- Evaluate likelihoods for parameter estimation
- Generate probability distributions for observational data

The analysis uses a cosmology with H0=70.0 km/s/Mpc, Om0=0.286, and processes
data at redshifts z=8.0, 12.0, and 16.0.
"""

import h5py
import numpy as np
import os.path as path
from argparse import ArgumentParser
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import xml.etree.ElementTree as ET
from joblib import Parallel, delayed
from numpy.typing import NDArray
from dataclasses import dataclass
import scipy
import scipy.stats
import scipy.optimize
from time import time


@dataclass
class Stats:
    """Statistics for the galaxy-halo connection.
    
    Stores statistical properties (mean/sigma, min, max) for both absolute
    and apparent magnitudes as a function of halo mass and redshift.
    
    Attributes
    ----------
    absolute_mean : NDArray
        Mean absolute magnitude, shape (nz, n_mass_bins).
    absolute_sigma : NDArray
        Standard deviation of absolute magnitude, shape (nz, n_mass_bins).
    absolute_min : NDArray
        Minimum absolute magnitude, shape (nz, n_mass_bins).
    absolute_max : NDArray
        Maximum absolute magnitude, shape (nz, n_mass_bins).
    apparent_mean : NDArray
        Mean apparent magnitude, shape (nz, n_mass_bins).
    apparent_sigma : NDArray
        Standard deviation of apparent magnitude, shape (nz, n_mass_bins).
    apparent_min : NDArray
        Minimum apparent magnitude, shape (nz, n_mass_bins).
    apparent_max : NDArray
        Maximum apparent magnitude, shape (nz, n_mass_bins).
    """
    absolute_mean: NDArray
    absolute_sigma: NDArray
    absolute_min: NDArray
    absolute_max: NDArray
    apparent_mean: NDArray
    apparent_sigma: NDArray
    apparent_min: NDArray
    apparent_max: NDArray


@dataclass
class SkewedStats:
    """Skewed statistics for the galaxy-halo connection.
    
    Stores statistical properties using medians and asymmetric standard deviations
    (left and right) for both absolute and apparent magnitudes. Used for
    two-sided normal distributions that better capture asymmetric distributions.
    
    Attributes
    ----------
    absolute_median : NDArray
        Median absolute magnitude, shape (nz, n_mass_bins).
    absolute_sigma_left : NDArray
        Standard deviation for absolute magnitude below median, shape (nz, n_mass_bins).
    absolute_sigma_right : NDArray
        Standard deviation for absolute magnitude above median, shape (nz, n_mass_bins).
    absolute_min : NDArray
        Minimum absolute magnitude, shape (nz, n_mass_bins).
    absolute_max : NDArray
        Maximum absolute magnitude, shape (nz, n_mass_bins).
    apparent_median : NDArray
        Median apparent magnitude, shape (nz, n_mass_bins).
    apparent_sigma_left : NDArray
        Standard deviation for apparent magnitude below median, shape (nz, n_mass_bins).
    apparent_sigma_right : NDArray
        Standard deviation for apparent magnitude above median, shape (nz, n_mass_bins).
    apparent_min : NDArray
        Minimum apparent magnitude, shape (nz, n_mass_bins).
    apparent_max : NDArray
        Maximum apparent magnitude, shape (nz, n_mass_bins).
    """
    absolute_median: NDArray
    absolute_sigma_left: NDArray
    absolute_sigma_right: NDArray
    absolute_min: NDArray
    absolute_max: NDArray
    apparent_median: NDArray
    apparent_sigma_left: NDArray
    apparent_sigma_right: NDArray
    apparent_min: NDArray
    apparent_max: NDArray

rng = np.random.default_rng()

# Cosmology used throughout the analysis
cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)


def get_weights_from_hmf(
    filename: str,
    load: bool = True,
    save: bool = True,
) -> NDArray:
    """Convert halo mass function (HMF) output to corresponding merger tree weights.
    
    Converts HMF output from Galacticus to merger tree weights for evaluating
    the HMF at intermediate redshifts. The weights are computed by integrating
    the HMF over halo mass bins.

    Parameters
    ----------
    filename : str
        Path to the input HMF HDF5 file from Galacticus.
    load : bool, optional
        If True, attempts to load pre-computed weights from 'data/hmf_weights.npy'
        instead of recomputing. Default is True.
    save : bool, optional
        If True, saves the computed weights to 'data/hmf_weights.npy'.
        Default is True.

    Returns
    -------
    NDArray
        Array of shape (n_redshifts, n_mass_bins) containing merger tree weights
        evaluated on a grid of redshift values. The number of redshifts is
        currently hardcoded to 17.

    Notes
    -----
    The halo mass range (1.0e8 to 5.0e11 Msun) and number of bins (3699) are
    hardcoded. The number of redshifts (17) is also hardcoded.
    """
    output_filename = 'data/hmf_weights.npy' 
    if load and path.isfile(output_filename):
        weights = np.load(output_filename) 
    else:
        weights = []
        f = h5py.File(filename,"r")
        # Hardcoded: halo mass range and number of bins
        halo_masses = np.geomspace(1.0e8, 5.0e11, 3699)
        log_halo_masses = np.log10(halo_masses)
        log_delta_mh = log_halo_masses[1] - log_halo_masses[0]
        left_bin_edge = 10**(log_halo_masses - log_delta_mh/2.0)
        right_bin_edge = 10**(log_halo_masses + log_delta_mh/2.0)
        bin_size = right_bin_edge - left_bin_edge
        weights = []
        # Hardcoded: number of redshifts (17)
        for i in range(17):
            j = 17-i
            output = f[f'Outputs/Output{j}']
            hmf = output['haloMassFunctionM'] 
            weight = bin_size * hmf
            weights.append(weight)
        weights = np.array(weights)
        f.close()
        if save:
            np.save(output_filename, weights)
    return weights


def get_data_from_hdf5(filename: str, jwst_filter_name: str) -> NDArray:
    """Extract log halo masses, magnitudes, and weights from Galacticus HDF5 file.

    Extracts galaxy properties from a Galacticus output HDF5 file, including
    halo masses, absolute and apparent magnitudes, redshifts, and merger tree
    weights. Data is sorted by halo mass.

    Parameters
    ----------
    filename : str
        Path to the Galacticus HDF5 output file.
    jwst_filter_name : str
        Name of the JWST filter for luminosities (e.g., 'JWST_NIRCAM_f277w').

    Returns
    -------
    NDArray
        Array of shape (n_galaxies, 5) containing:
        - Column 0: log10(halo mass) [Msun]
        - Column 1: apparent magnitude
        - Column 2: absolute magnitude
        - Column 3: redshift
        - Column 4: merger tree weight
    """
    outfile = h5py.File(filename, "r")
    outputs = outfile['Outputs']    
    node_data = outputs['Output1']['nodeData']
    halo_masses = node_data['basicMass'][:]
    tree_weights = node_data['mergerTreeWeight'][:]
    z = node_data['redshift'][0]
    sfilename = f'spheroidLuminositiesStellar:{jwst_filter_name}:observed:z{z:.4f}'
    dfilename = f'diskLuminositiesStellar:{jwst_filter_name}:observed:z{z:.4f}'
    
    luminosity = node_data[sfilename][:] + node_data[dfilename][:] 
    # Set lower limit on luminosity to prevent log(0)
    luminosity[luminosity<1.0e-10] = 1.0e-10 
    absolute_mag = -2.5 * np.log10(luminosity)
    apparent_mag = absolute_mag + cosmo.distmod(z).value - 2.5*np.log10(1+z)
    sorted_indices = np.argsort(halo_masses)
    halo_masses = halo_masses[sorted_indices]
    apparent_mag = apparent_mag[sorted_indices]
    absolute_mag = absolute_mag[sorted_indices]
    tree_weights = tree_weights[sorted_indices]
    redshifts = np.tile(z, len(halo_masses)) 
    data = np.stack((np.log10(halo_masses), apparent_mag, absolute_mag, redshifts, tree_weights),axis=-1)
    outfile.close()
    return data


def get_ngdeep_completeness(apparent_grid: NDArray) -> NDArray:
    """Estimate the completeness function for the NGDEEP survey.
    
    Computes the completeness function based on effective volumes from the
    NGDEEP survey. Uses data from Table 2 of arXiv:2306.06244 at z~9, normalized
    by the largest effective volume, and linearly interpolates between magnitude
    bins. Completeness is set to 0 at the 5-sigma limiting depth (30.4 mag).

    Parameters
    ----------
    apparent_grid : NDArray
        Array of apparent magnitudes on which to evaluate the completeness
        function.

    Returns
    -------
    NDArray
        Array with the completeness function (0-1) interpolated onto the
        input apparent magnitude grid.

    References
    ----------
    .. [1] https://arxiv.org/pdf/2306.06244 (NGDEEP survey data)
    """
    survey_absolute_magnitudes = np.array([-21.1, -20.1, -19.1, -18.35, -17.85, -17.35])
    survey_apparent_magnitudes = (survey_absolute_magnitudes + 
                                  cosmo.distmod(9.0).value - 2.5*np.log10(1+9.0))
    effective_volumes = np.array([18700., 18500., 15800., 13100., 7770., 2520.]) # Mpc^3
  
    completeness = effective_volumes / np.amax(effective_volumes) 
    # Set completeness to 0 at 5-sigma limiting depth (30.4 mag)
    apparent_magnitudes = np.append(survey_apparent_magnitudes, 30.4)
    completeness = np.append(completeness, 0.0)    
    # Extend values to grid min/max for interpolation
    completeness = np.concatenate(([1.0], completeness, [0.0]))
    grid_min = np.amin(apparent_grid)
    grid_max = np.amax(apparent_grid)
    apparent_magnitudes = np.concatenate(([grid_min], apparent_magnitudes, 
                                          [grid_max]))
    completeness = np.interp(apparent_grid, apparent_magnitudes, completeness)
    return completeness


def get_ceers_completeness(apparent_grid: NDArray) -> NDArray:
    """Estimate the completeness function for the CEERS survey.
    
    Computes the completeness function based on effective volumes from the
    CEERS survey. Uses data from Table 4 of arXiv:2311.04279 at z~9, normalized
    by the largest effective volume, and linearly interpolates between magnitude
    bins. Completeness is set to 0 at the 5-sigma limiting depth (29.15 mag).

    Parameters
    ----------
    apparent_grid : NDArray
        Array of apparent magnitudes on which to evaluate the completeness
        function.

    Returns
    -------
    NDArray
        Array with the completeness function (0-1) interpolated onto the
        input apparent magnitude grid.

    References
    ----------
    .. [1] https://arxiv.org/pdf/2311.04279 (CEERS survey data)
    """
    survey_absolute_magnitudes = np.array([-22.5, -22.0, -21.5, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5])
    survey_apparent_magnitudes = (survey_absolute_magnitudes + 
                                  cosmo.distmod(9.0).value - 2.5*np.log10(1+9.0))
    effective_volumes = np.array([187000., 187000., 187000., 193000., 177000., 
                                  161000., 120000., 77900., 18600.]) # Mpc^3
    completeness = effective_volumes / np.amax(effective_volumes) 
    # Set completeness to 0 at 5-sigma limiting depth (29.15 mag)
    apparent_magnitudes = np.append(survey_apparent_magnitudes, 29.15)
    completeness = np.append(completeness, 0.0)
    # Extend values to grid min/max for interpolation
    completeness = np.concatenate(([1.0], completeness, [0.0]))
    grid_min = np.amin(apparent_grid)
    grid_max = np.amax(apparent_grid)
    apparent_magnitudes = np.concatenate(([grid_min], apparent_magnitudes, 
                                          [grid_max]))    
    cf = np.interp(apparent_grid, apparent_magnitudes, completeness)
    return cf


def get_data_pdf(
    observed_data: pd.DataFrame, 
    apparent_grid: NDArray, 
    redshift_grid: NDArray, 
    filename: str, 
    load: bool = True,
    save: bool = True
) -> NDArray:
    """Compute probability density functions for galaxy candidates from redshift uncertainties.
    
    Computes a probability density function for each galaxy candidate assuming
    a two-sided normal distribution using the median redshifts and one-sigma
    uncertainties. A lower redshift limit at z=8.5 is applied to mimic color
    cuts used in high-z galaxy candidate selection, after which the PDFs are
    renormalized.

    Parameters
    ----------
    observed_data : pd.DataFrame
        DataFrame with columns:
        - 'mf277w': apparent magnitudes
        - 'z': median redshifts
        - 'z_upper_err': one-sigma error above the median
        - 'z_lower_err': one-sigma error below the median
    apparent_grid : NDArray
        Grid of apparent magnitudes on which the PDF is evaluated.
    redshift_grid : NDArray
        Grid of redshifts on which the PDF is evaluated.
    filename : str
        Filename to save/load the computed PDFs.
    load : bool, optional
        If True, loads pre-computed PDFs from filename. Default is True.
    save : bool, optional
        If True, saves the computed PDFs to filename. Default is True.

    Returns
    -------
    NDArray
        Array of shape (n_gal, n_mag, n_z) containing PDFs for each galaxy
        candidate, where:
        - n_gal: number of galaxies in observed_data
        - n_mag: length of apparent_grid
        - n_z: length of redshift_grid

    References
    ----------
    .. [1] https://arxiv.org/pdf/2306.06244 (NGDEEP)
    .. [2] https://arxiv.org/pdf/2311.04279 (CEERS)
    """
    z_cutoff = 8.5 
    if load and path.isfile(filename):
        obs_pdf = np.load(filename)
    else:
        n_gal = len(observed_data)
        n_mag = len(apparent_grid)
        n_z = len(redshift_grid)
        obs_pdf = np.zeros((n_gal,n_mag,n_z))
        for i in range(n_gal):
            mag = observed_data['mf277w'][i]
            z = observed_data['z'][i]
            z_upper_err = observed_data['z_upper_err'][i]
            z_lower_err = np.abs(observed_data['z_lower_err'][i])
            dz = redshift_grid[1]-redshift_grid[0]
            
            mag_pdf = np.zeros_like(apparent_grid)
            idx = np.argmin(np.abs(apparent_grid-mag))
            mag_pdf[idx] = 1.0
            mag_pdf = mag_pdf.reshape(-1, 1)
            
            # Two-sided normal: different sigmas for z < median and z >= median
            lower_idx = redshift_grid <= z
            upper_idx = redshift_grid > z
            z_pdf = np.zeros_like(redshift_grid)
            norm = np.sqrt(2.0/np.pi)/(z_upper_err + z_lower_err)
            z_pdf[lower_idx] = norm * np.exp(-(redshift_grid[lower_idx]-z)**2 / 2.0 / z_lower_err**2)
            z_pdf[upper_idx] = norm * np.exp(-(redshift_grid[upper_idx]-z)**2 / 2.0 / z_upper_err**2)
            z_pdf = z_pdf.reshape(1, -1)
            
            obs_pdf[i,:,:] = mag_pdf * z_pdf * dz 
 
            # Apply z=8.5 cutoff to mimic color cuts used in candidate selection
            cut_idx = redshift_grid < 8.5
            obs_pdf[i][:,cut_idx] = 0
            new_norm = np.sum(obs_pdf[i,:,:])
            obs_pdf[i,:,:] = obs_pdf[i,:,:] / new_norm  # Renormalize
        if save:
            np.save(filename, obs_pdf)
    return obs_pdf

    
def load_data(
    data_directory: str, 
    reload: bool = True, 
    save: bool = True,
) -> pd.DataFrame:
    """Load data for one parameter combination from Galacticus HDF5 files.
    
    Loads the three HDF5 files corresponding to redshifts z=8.0, 12.0, and 16.0
    and combines them into a single DataFrame. Data can be cached to CSV for
    faster subsequent loads.

    Parameters
    ----------
    data_directory : str
        Path to the directory containing the HDF5 files (z8.0.hdf5, z12.0.hdf5,
        z16.0.hdf5).
    reload : bool, optional
        If True, forces reload from HDF5 files even if CSV cache exists.
        Default is True.
    save : bool, optional
        If True, saves the combined data to 'data_directory/data.csv' for
        faster future loads. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'log_halo_mass': log10(halo mass) [Msun]
        - 'apparent_magnitude': apparent magnitude
        - 'absolute_magnitude': absolute magnitude
        - 'redshift': redshift
        - 'tree_weight': merger tree weight
    """ 
    data_columns= ['log_halo_mass', 'apparent_magnitude', 'absolute_magnitude', 'redshift', 'tree_weight']
    data_filename = path.join(data_directory, 'data.csv')
    if not reload and path.isfile(data_filename):
        data = pd.read_csv(data_filename)
    else:
        data_list = []
        for z in ['8.0', '12.0', '16.0']: 
            hdf5_filename = path.join(data_directory,f'z{z}.hdf5')
            data = get_data_from_hdf5(hdf5_filename, 'JWST_NIRCAM_f277w')
            data_list.append(data)
        data = np.concatenate(data_list, axis=0)
        data = pd.DataFrame(data, columns=data_columns)
        if save:
            data.to_csv(data_filename, index=False)
    return data


def get_binned_weights(
    logmh_bins: NDArray, 
    logmhs: NDArray, 
    weights: NDArray, 
    load: bool = True,
    save: bool = True,
) -> NDArray:
    """Calculate binned merger tree weights.

    Bins merger tree weights by halo mass for each redshift. The weights are
    summed within each mass bin to produce a binned weight array.

    Parameters
    ----------
    logmh_bins : NDArray
        Array of bin edges for log halo mass bins.
    logmhs : NDArray
        Array of unique log halo masses (will be made unique if not already).
    weights : NDArray
        Array of shape (n_redshifts, n_halos) containing merger tree weights
        for each halo at each redshift.
    load : bool, optional
        If True, loads pre-computed weights from 'data/binned_zgrid_weights.npy'.
        Default is True.
    save : bool, optional
        If True, saves computed weights to 'data/binned_zgrid_weights.npy'.
        Default is True.
        
    Returns
    -------
    NDArray
        Array of shape (n_redshifts, n_mass_bins) containing binned merger
        tree weights.

    Notes
    -----
    This function uses global variables `nz` and `n_mass_bins` which must be
    defined in the module scope.
    """
    output_filename =  'data/binned_zgrid_weights.npy'
    if load and path.isfile(output_filename):
        binned_weights = np.load(output_filename)
    else:
        logmhs = np.unique(logmhs)
        binned_weights = np.zeros((nz,n_mass_bins))
        for i in range(n_mass_bins):
            left_bin_edge = Mh_bins[i]
            right_bin_edge = Mh_bins[i+1]
            bin_index = (logmhs > left_bin_edge) & (logmhs < right_bin_edge)
            binned_weights[:,i] = np.sum(weights[:,bin_index],axis=1)
        if save:
            np.save(output_filename, binned_weights)
    return binned_weights


def get_probs(
    magnitude_grid: NDArray, 
    stats: Stats, 
    data_directory: str, 
    do_abs: bool, 
    recompute: bool,
    save: bool=True,
) -> NDArray:
    """Compute probability distributions for the galaxy-halo connection.

    Computes probability density functions (PDFs) for galaxy magnitudes as a
    function of halo mass and redshift. The PDF is determined by a normal
    distribution with mean and standard deviation from simulated data. The PDF
    is cut off at fractions of the min/max magnitudes to prevent artificial
    upscatter of faint galaxies.

    Parameters
    ----------
    magnitude_grid : NDArray
        Grid of magnitudes (absolute or apparent) on which to evaluate PDFs.
    stats : Stats
        Dataclass instance containing statistics (mean, sigma, min, max) calculated
        from the simulated data. Can be None if recompute is False and file
        exists.
    data_directory : str
        Directory where PDF files are saved/loaded.
    do_abs : bool
        If True, compute PDFs for absolute magnitudes; if False, for apparent
        magnitudes.
    recompute : bool
        If True, forces recomputation even if PDF file exists.
    save : bool, optional
        If True, saves the computed PDFs to file. Default is True.
    
    Returns
    -------
    NDArray
        Array of shape (n_mag, n_z, n_mass_bins) containing probability
        distributions. PDFs are normalized to sum to 1 over the magnitude
        dimension.

    Notes
    -----
    - For absolute magnitudes: cutoff at 1.1 * min and 0.9 * max
    - For apparent magnitudes: cutoff at 0.95 * min and 1.05 * max
    - Uses global variables `nz` and `n_mass_bins` which must be defined in
      module scope.
    """
    if do_abs:
        if stats is not None:
            mins = stats.absolute_min
            maxs = stats.absolute_max
            mean = stats.absolute_mean
            std = stats.absolute_sigma
            probability_min_magnitude = mins * 1.1 # absolute mags are negative
            probability_max_magnitude = maxs * 0.9
        probs_filename = path.join(data_directory, 'absolute_pdf.npy')
        # Cutoff PDF at faint end to prevent artificial upscatter
        # (distribution becomes non-Gaussian; values from empirical testing)
        
    else:
        if stats is not None:
            mins = stats.apparent_min
            maxs = stats.apparent_max
            mean = stats.apparent_mean
            std = stats.apparent_sigma
            probability_min_magnitude = mins * 0.95
            probability_max_magnitude = maxs * 1.05 
        probs_filename = path.join(data_directory, 'apparent_pdf.npy')
        # Cutoff PDF at faint end to prevent artificial upscatter
        # (distribution becomes non-Gaussian; values from empirical testing) 
        
    if path.isfile(probs_filename) and not recompute:
        pdfs = np.load(probs_filename)
    else:
        magnitude_grid = magnitude_grid.reshape(-1,1,1)
        mean = mean.reshape(1, nz, n_mass_bins)
        std = std.reshape(1, nz, n_mass_bins)
        pdfs = np.exp(-(magnitude_grid-mean)**2/2.0/std**2)/np.sqrt(2*np.pi)/std
        
        for i, mag in enumerate(magnitude_grid):
            idx = (mag > probability_max_magnitude) | (mag < probability_min_magnitude)
            pdfs[i][idx] = 0

        # Renormalize after cutting off the PDF
        norm = np.sum(pdfs, axis=0) 
        norm = norm.reshape(1, nz, n_mass_bins)        
        pdfs = (pdfs / norm)

        if save:
            np.save(probs_filename, pdfs)
    return pdfs


def get_skewed_probs(
    magnitude_grid: NDArray, 
    stats: SkewedStats, 
    data_directory: str, 
    do_abs: bool, 
    recompute: bool,
    save: bool=True,
) -> NDArray:
    """Compute skewed probability distributions for the galaxy-halo connection.

    Similar to `get_probs`, but uses a two-sided normal distribution (skewed
    PDF) with different standard deviations on the left and right sides of the
    median. This better captures asymmetric distributions in the galaxy-halo
    connection.

    Parameters
    ----------
    magnitude_grid : NDArray
        Grid of magnitudes (absolute or apparent) on which to evaluate PDFs.
    stats : SkewedStats
        Dataclass containing statistics (median, sigma_left, sigma_right,
        min, max) calculated from the simulated data. Can be None if recompute
        is False and file exists.
    data_directory : str
        Directory where PDF files are saved/loaded.
    do_abs : bool
        If True, compute PDFs for absolute magnitudes; if False, for apparent
        magnitudes.
    recompute : bool
        If True, forces recomputation even if PDF file exists.
    save : bool, optional
        If True, saves the computed PDFs to file. Default is True.
    
    Returns
    -------
    NDArray
        Array of shape (n_mag, n_z, n_mass_bins) containing skewed probability
        distributions. PDFs are normalized to sum to 1 over the magnitude
        dimension.

    Notes
    -----
    - Uses `two_sided_normal_pdf` for the skewed distribution
    - Cutoff fractions same as `get_probs`
    - Uses global variables `nz` and `n_mass_bins`
    """
    if do_abs:
        if stats is not None: 
            mu = stats.absolute_median
            sigma_L = stats.absolute_sigma_left
            sigma_R = stats.absolute_sigma_right
            probability_min_magnitude = stats.absolute_min * 1.1
            probability_max_magnitude = stats.absolute_max * 0.9
        probs_filename = path.join(data_directory, 'skewed_absolute_pdf.npy')
    else:
        if stats is not None:
            mu = stats.apparent_median
            sigma_L = stats.apparent_sigma_left
            sigma_R = stats.apparent_sigma_right
            probability_min_magnitude = stats.apparent_min * 0.95
            probability_max_magnitude = stats.apparent_max * 1.05 
        probs_filename = path.join(data_directory, 'skewed_apparent_pdf.npy')
    if path.isfile(probs_filename) and not recompute:
        pdfs = np.load(probs_filename)
    else:
        pdfs = np.zeros((len(magnitude_grid), nz, n_mass_bins))
        for i in range(nz):
            for j in range(n_mass_bins):
                pdfs[:,i,j] = two_sided_normal_pdf(magnitude_grid, mu[i,j], 
                                                   sigma_L[i,j], sigma_R[i,j])
        # Apply magnitude cutoffs
        for i, mag in enumerate(magnitude_grid):
            idx = (mag > probability_max_magnitude) | (mag < probability_min_magnitude)
            pdfs[i][idx] = 0
        # Renormalize after cutting off the PDF
        norm = np.sum(pdfs, axis=0) 
        norm = norm.reshape(1, nz, n_mass_bins)        
        pdfs = (pdfs / norm)

        if save:
            np.save(probs_filename, pdfs)
    return pdfs


def get_uvlf(
    probs: NDArray, 
    binned_weights: NDArray,
    data_directory: str, 
    do_abs: bool, 
    do_skewed: bool,
    recompute: bool,
) -> NDArray:
    """Compute the UV luminosity function (UVLF) from galaxy-halo PDFs and weights.

    Computes the UV luminosity function by convolving the galaxy-halo connection
    probability distributions with the binned merger tree weights. The UVLF
    represents the number density of galaxies as a function of magnitude and
    redshift.

    Parameters
    ----------
    probs : NDArray
        Array of shape (n_mag, n_z, n_mass_bins) containing probability
        distributions from `get_probs` or `get_skewed_probs`.
    binned_weights : NDArray
        Array of shape (n_z, n_mass_bins) containing binned merger tree weights.
    data_directory : str
        Directory where UVLF files are saved/loaded.
    do_abs : bool
        If True, compute UVLF for absolute magnitudes; if False, for apparent
        magnitudes.
    do_skewed : bool
        If True, uses 'skewed_' prefix in output filename.
    recompute : bool
        If True, forces recomputation even if UVLF file exists.

    Returns
    -------
    NDArray
        Array of shape (n_mag, n_z) containing the UV luminosity function.
        Units depend on the input weights and magnitude grid spacing.
    """  
    if do_skewed:
        prefix = 'skewed_'
    else:
        prefix = ''
    if do_abs:
        uvlf_filename = path.join(data_directory, prefix+'absolute_uvlf.npy')
    else:
        uvlf_filename = path.join(data_directory, prefix+'apparent_uvlf.npy')
    if path.isfile(uvlf_filename) and not recompute:
        uvlf = np.load(uvlf_filename)
    else:
        uvlf = np.sum(binned_weights * probs, axis=2)
        np.save(uvlf_filename, uvlf)
    return uvlf




def calculate_likelihood(apparent_uvlf: NDArray) -> float:
    """Calculate the log-likelihood function given the apparent magnitude UVLF.
    
    Computes the Poisson log-likelihood comparing the model UVLF predictions
    with observed galaxy counts from NGDEEP and CEERS surveys. The likelihood
    accounts for both the expected number of galaxies and the probability of
    observing each candidate galaxy given redshift uncertainties.

    Parameters
    ----------
    apparent_uvlf : NDArray
        UV luminosity function as a function of apparent magnitude and redshift.
        Shape should be (n_mag, n_z).

    Returns
    -------
    float
        Log-likelihood value. Higher values indicate better agreement with
        observations.

    Notes
    -----
    - Only uses redshifts >= 8.5 (z_cutoff)
    - Uses global variables: redshift_grid, ngdeep_pdf, ceers_pdf,
      ngdeep_effective_volume, ceers_effective_volume
    - See arXiv:2410.11680 for theoretical details

    References
    ----------
    .. [1] https://arxiv.org/abs/2410.11680
    """
    z_cutoff = 8.5
    zidx = redshift_grid >= z_cutoff
    ngdeep_n = np.sum(apparent_uvlf[:,zidx]*ngdeep_effective_volume[:,zidx], dtype=np.longdouble) 
    ceers_n = np.sum(apparent_uvlf[:,zidx]*ceers_effective_volume[:,zidx], dtype=np.longdouble) 
    log_likelihood = -ngdeep_n-ceers_n
    ngdeep_obs_n = np.sum(ngdeep_pdf[:,:,zidx]*apparent_uvlf[:,zidx]*ngdeep_effective_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
    ngdeep_obs_n = np.log(ngdeep_obs_n[ngdeep_obs_n>0])
    ngdeep_obs_n = np.sum(ngdeep_obs_n)
    log_likelihood += ngdeep_obs_n
    ceers_obs_n = np.sum(ceers_pdf[:,:,zidx]*apparent_uvlf[:,zidx]*ceers_effective_volume[:,zidx], axis=(1,2), dtype=np.longdouble) 
    ceers_obs_n = np.log(ceers_obs_n[ceers_obs_n>0])
    ceers_obs_n = np.sum(ceers_obs_n)
    log_likelihood += ceers_obs_n
    return log_likelihood


# these are globals to save a little bit of time when running the calculations
# should probably get shoved into another file...
ngdeep_data = pd.read_csv('data/ngdeep_data.csv')
ngdeep_data = ngdeep_data[ngdeep_data['mf277w'] < 30.4]
ngdeep_data.reset_index(inplace=True)
ceers_data = pd.read_csv('data/CEERS_data.csv')
ceers_data = ceers_data[ceers_data['mf277w'] < 29.15]
ceers_data.reset_index(inplace=True)
zgrid_weights = get_weights_from_hmf('data/zgrid_hmfs.hdf5')

dMh = 0.15
Mh_bins = np.arange(8.0, 11.76, dMh)
global n_mass_bins
n_mass_bins = len(Mh_bins)-1
bin_centers = Mh_bins[:-1] + dMh/2.0 

global nz
nz = 17
global redshift_grid
redshift_grid = np.linspace(8.0, 16.0, nz)
dz = redshift_grid[1]-redshift_grid[0]

cv_z = [7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0]
cv = [0.1522, 0.17411, 0.18843, 0.22613, 0.2549, 0.26993, 0.292834, 0.3769, 0.426021, 0.43414, 0.49211, 0.59527, 0.664436, 0.62814, 0.602, 0.602]
cv = np.interp(redshift_grid, cv_z, cv)

logmhs = np.log10(np.geomspace(1.0e8,5.0e11,3699))
global binned_weights
binned_weights = get_binned_weights(Mh_bins,logmhs,zgrid_weights, True)

abs_min = -25.0
abs_max = 0.
dabs = 0.2
nabs = int(round((abs_max-abs_min)/dabs))+1
absolute_magnitude_grid = np.linspace(abs_min, abs_max, nabs) 
dabs = absolute_magnitude_grid[1]-absolute_magnitude_grid[0]

app_min = 22.0
app_max = 45.0
dapp = 0.25 
napp = int(round((app_max-app_min)/dapp))+1
apparent_magnitude_grid = np.linspace(app_min, app_max, napp) 
dapp = apparent_magnitude_grid[1]-apparent_magnitude_grid[0]

# global t
t_grid = cosmo.age(redshift_grid).value
t8 = cosmo.age(8.0).value
t12 = cosmo.age(12.0).value
t16 = cosmo.age(16.0).value

z_volumes = (cosmo.comoving_volume(redshift_grid+dz/2.0)-cosmo.comoving_volume(redshift_grid-dz/2.0)).value 

ngdeep_cf = get_ngdeep_completeness(apparent_magnitude_grid)
ngdeep_area = 3.3667617763401435e-08 # 5 arcmin^2 as a fraction of the sky
ngdeep_volumes = ngdeep_area * z_volumes  # Differential comoving volume per redshift
ngdeep_effective_volume = ngdeep_cf.reshape(-1,1)*ngdeep_volumes.reshape(1,-1)
ngdeep_effective_volume = np.abs(ngdeep_effective_volume)

ceers_cf = get_ceers_completeness(apparent_magnitude_grid)
ceers_area = 5.932234249911333e-07 # 88.1 arcmin^2 as a fraction of the sky
ceers_volumes = ceers_area * z_volumes # Differential comoving volume per redshift per steradian at each input redshift.
ceers_effective_volume = ceers_cf.reshape(-1,1)*ceers_volumes.reshape(1,-1)
ceers_effective_volume = np.abs(ceers_effective_volume)

ngdeep_pdf_filename =  'data/ngdeep_pdf.npy'
ceers_pdf_filename = 'data/ceers_pdf.npy'
ngdeep_pdf = np.abs(get_data_pdf(ngdeep_data, apparent_magnitude_grid, redshift_grid, ngdeep_pdf_filename, True))
ceers_pdf = np.abs(get_data_pdf(ceers_data, apparent_magnitude_grid, redshift_grid, ceers_pdf_filename, True))


def two_sided_normal_pdf(x: NDArray, mu: float, sigma_L: float, sigma_R: float) -> NDArray:
    """Compute a two-sided normal (skewed) probability density function.
    
    A PDF with different standard deviations on the left and right sides of
    the mean, allowing for asymmetric distributions.

    Parameters
    ----------
    x : NDArray
        Points at which to evaluate the PDF.
    mu : float
        Mean/median of the distribution.
    sigma_L : float
        Standard deviation for x < mu (left side).
    sigma_R : float
        Standard deviation for x >= mu (right side).

    Returns
    -------
    NDArray
        Probability density values at each point in x.
    """
    return np.where(x < mu,
                    scipy.stats.norm.pdf(x, loc=mu, scale=sigma_L),
                    scipy.stats.norm.pdf(x, loc=mu, scale=sigma_R))


def neg_log_likelihood(params: tuple, x: NDArray) -> float:
    """Negative log-likelihood for fitting a two-sided normal distribution.
    
    Used as an objective function for optimization when fitting skewed
    distributions to data.

    Parameters
    ----------
    params : tuple
        Parameters (mu, sigma_L, sigma_R) of the two-sided normal distribution.
    x : NDArray
        Data points to fit.

    Returns
    -------
    float
        Negative log-likelihood value (to be minimized).
    """
    mu, sigma_L, sigma_R = params
    sigma_L, sigma_R = abs(sigma_L), abs(sigma_R)  # Ensure positive values
    
    likelihoods = two_sided_normal_pdf(x, mu, sigma_L, sigma_R)
    return -np.sum(np.log(likelihoods + 1e-10)) 


def get_skewed_stats(data: pd.DataFrame) -> SkewedStats:
    """Calculate skewed statistics (medians and asymmetric sigmas) from galaxy data.
    
    Computes statistics for the galaxy-halo connection using a two-sided normal
    distribution. For each halo mass bin and redshift, fits a skewed distribution
    to the magnitude data and extracts median, left/right standard deviations,
    and min/max values. Interpolates between the three sampled redshifts (z=8,
    12, 16) to the full redshift grid.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing galaxy data with columns: 'redshift',
        'log_halo_mass', 'absolute_magnitude', 'apparent_magnitude'.

    Returns
    -------
    SkewedStats
        Dataclass instance containing:
        - absolute_median: (nz, n_mass_bins) array
        - absolute_sigma_left: (nz, n_mass_bins) array
        - absolute_sigma_right: (nz, n_mass_bins) array
        - absolute_min: (nz, n_mass_bins) array
        - absolute_max: (nz, n_mass_bins) array
        - apparent_median: (nz, n_mass_bins) array
        - apparent_sigma_left: (nz, n_mass_bins) array
        - apparent_sigma_right: (nz, n_mass_bins) array
        - apparent_min: (nz, n_mass_bins) array
        - apparent_max: (nz, n_mass_bins) array

    Notes
    -----
    - Uses scipy.optimize.minimize with L-BFGS-B method to fit distributions
    - Interpolates in cosmic time (age) rather than redshift
    - Uses global variables: nz, n_mass_bins, Mh_bins, t_grid, t8, t12, t16
    """
    sampled_redshifts = np.unique(data['redshift'])[::-1]
    sampled_nz = len(sampled_redshifts)
    sampled_absolute_medians = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_sigma_lefts = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_sigma_rights = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_mins = np.zeros((sampled_nz, n_mass_bins))

    sampled_apparent_medians = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_sigma_lefts = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_sigma_rights = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_mins = np.zeros((sampled_nz, n_mass_bins))
    for i,z in enumerate(sampled_redshifts):
        zidx = data['redshift']==z
        z_lmh = data['log_halo_mass'][zidx]
        z_abs = data['absolute_magnitude'][zidx]
        z_app = data['apparent_magnitude'][zidx]

        for j in range(n_mass_bins):
            left = Mh_bins[j]
            right = Mh_bins[j+1]
            bidx = (z_lmh > left) & (z_lmh < right)

            abs_samples = z_abs[bidx]
            mu_init = np.median(abs_samples)
            sigma_L_init = np.std(abs_samples[abs_samples < mu_init])
            sigma_R_init = np.std(abs_samples[abs_samples >= mu_init])
            abs_result = scipy.optimize.minimize(
                neg_log_likelihood, 
                x0=[mu_init, sigma_L_init, sigma_R_init], 
                args=(abs_samples,),
                method='L-BFGS-B', 
                bounds=[(-30, 0), (1e-2, 1e2), (1e-2, 1e2)]
                )
            sampled_absolute_medians[i,j] = abs_result.x[0]
            sampled_absolute_sigma_lefts[i,j] = abs_result.x[1]
            sampled_absolute_sigma_rights[i,j] = abs_result.x[2]
            sampled_absolute_maxs[i,j] = np.amax(abs_samples)
            sampled_absolute_mins[i,j] = np.amin(abs_samples)

            app_samples = z_app[bidx]
            mu_init = np.median(app_samples)
            sigma_L_init = np.std(app_samples[app_samples < mu_init])
            sigma_R_init = np.std(app_samples[app_samples >= mu_init])
            app_result = scipy.optimize.minimize(
                neg_log_likelihood, 
                x0=[mu_init, sigma_L_init, sigma_R_init], 
                args=(app_samples,),
                method='L-BFGS-B', 
                bounds=[(0, 50), (1e-2, 1e2), (1e-2, 1e2)]
                )
            sampled_apparent_medians[i,j] = app_result.x[0]
            sampled_apparent_sigma_lefts[i,j] = app_result.x[1]
            sampled_apparent_sigma_rights[i,j] = app_result.x[2] 
            sampled_apparent_maxs[i,j] = np.amax(app_samples)
            sampled_apparent_mins[i,j] = np.amin(app_samples)        

    absolute_medians = np.zeros((nz, n_mass_bins))
    absolute_sigma_lefts = np.zeros((nz, n_mass_bins))
    absolute_sigma_rights = np.zeros((nz, n_mass_bins))
    absolute_maxs = np.zeros((nz, n_mass_bins))
    absolute_mins = np.zeros((nz, n_mass_bins))

    apparent_medians = np.zeros((nz, n_mass_bins))
    apparent_sigma_lefts = np.zeros((nz, n_mass_bins))
    apparent_sigma_rights = np.zeros((nz, n_mass_bins))
    apparent_maxs = np.zeros((nz, n_mass_bins))
    apparent_mins = np.zeros((nz, n_mass_bins))

    sampled_times = [t16, t12, t8]
    for j in range(n_mass_bins):
        absolute_medians[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_medians[:,j])
        absolute_sigma_lefts[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_sigma_lefts[:,j])
        absolute_sigma_rights[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_sigma_rights[:,j])
        absolute_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_maxs[:,j])
        absolute_mins[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_mins[:,j])

        apparent_medians[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_medians[:,j])
        apparent_sigma_lefts[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_sigma_lefts[:,j])
        apparent_sigma_rights[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_sigma_rights[:,j])
        apparent_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_maxs[:,j])
        apparent_mins[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_mins[:,j])

    stats = SkewedStats(
        absolute_median=absolute_medians,
        absolute_sigma_left=absolute_sigma_lefts,
        absolute_sigma_right=absolute_sigma_rights,
        absolute_min=absolute_mins,
        absolute_max=absolute_maxs,
        apparent_median=apparent_medians,
        apparent_sigma_left=apparent_sigma_lefts,
        apparent_sigma_right=apparent_sigma_rights,
        apparent_min=apparent_mins,
        apparent_max=apparent_maxs
    )
    return stats


def get_stats(data: pd.DataFrame) -> Stats:
    """Calculate statistics (means and standard deviations) from galaxy data.
    
    Computes statistics for the galaxy-halo connection using normal distributions.
    For each halo mass bin and redshift, calculates mean, standard deviation,
    min, and max for both absolute and apparent magnitudes. Interpolates
    between the three sampled redshifts (z=8, 12, 16) to the full redshift grid.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing galaxy data with columns: 'redshift',
        'log_halo_mass', 'absolute_magnitude', 'apparent_magnitude'.

    Returns
    -------
    Stats
        Dataclass instance containing:
        - absolute_means: (nz, n_mass_bins) array
        - absolute_sigmas: (nz, n_mass_bins) array
        - absolute_mins: (nz, n_mass_bins) array
        - absolute_maxs: (nz, n_mass_bins) array
        - apparent_means: (nz, n_mass_bins) array
        - apparent_sigmas: (nz, n_mass_bins) array
        - apparent_mins: (nz, n_mass_bins) array
        - apparent_maxs: (nz, n_mass_bins) array

    Notes
    -----
    - Interpolates in cosmic time (age) rather than redshift
    - Apparent magnitude means are computed from absolute means using distance
      modulus and k-correction
    - Uses global variables: nz, n_mass_bins, Mh_bins, t_grid, t8, t12, t16,
      redshift_grid, cosmo
    """
    sampled_redshifts = np.unique(data['redshift'])[::-1]
    sampled_nz = len(sampled_redshifts)
    sampled_absolute_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_mins = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_means = np.zeros((sampled_nz, n_mass_bins))
    sampled_absolute_stds = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_maxs = np.zeros((sampled_nz, n_mass_bins))
    sampled_apparent_mins = np.zeros((sampled_nz, n_mass_bins))
    for i,z in enumerate(sampled_redshifts):
        zidx = data['redshift']==z
        z_lmh = data['log_halo_mass'][zidx]
        z_abs = data['absolute_magnitude'][zidx]
        z_app = data['apparent_magnitude'][zidx]

        for j in range(n_mass_bins):
            left = Mh_bins[j]
            right = Mh_bins[j+1]
            bidx = (z_lmh > left) & (z_lmh < right)
            sampled_absolute_means[i,j] = np.average(z_abs[bidx])
            sampled_absolute_stds[i,j] = np.std(z_abs[bidx])
            sampled_absolute_maxs[i,j] = np.amax(z_abs[bidx])
            sampled_absolute_mins[i,j] = np.amin(z_abs[bidx])
            sampled_apparent_maxs[i,j] = np.amax(z_app[bidx])
            sampled_apparent_mins[i,j] = np.amin(z_app[bidx])

    absolute_sigmas = np.zeros((nz, n_mass_bins))
    absolute_means = np.zeros((nz, n_mass_bins))
    absolute_maxs = np.zeros((nz, n_mass_bins))
    absolute_mins = np.zeros((nz, n_mass_bins))
    apparent_maxs = np.zeros((nz, n_mass_bins))
    apparent_mins = np.zeros((nz, n_mass_bins))

    sampled_times = [t16, t12, t8]
    for j in range(n_mass_bins):
        absolute_means[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_means[:,j])
        absolute_sigmas[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_stds[:,j])
        absolute_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_maxs[:,j])
        absolute_mins[:,j] = np.interp(t_grid, sampled_times, sampled_absolute_mins[:,j])
        apparent_maxs[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_maxs[:,j])
        apparent_mins[:,j] = np.interp(t_grid, sampled_times, sampled_apparent_mins[:,j])
    shift = cosmo.distmod(redshift_grid).value-2.5*np.log10(1.+redshift_grid)
    apparent_means = absolute_means + shift.reshape(-1,1)
    apparent_sigmas = absolute_sigmas.copy()
    stats = Stats(
        absolute_mean=absolute_means,
        absolute_sigma=absolute_sigmas,
        absolute_min=absolute_mins,
        absolute_max=absolute_maxs,
        apparent_mean=apparent_means,
        apparent_sigma=apparent_sigmas,
        apparent_min=apparent_mins,
        apparent_max=apparent_maxs
    )
    return stats


def evaluate_likelihood(i: int, data_directory: str, skewed: bool, 
                       reload: bool, recompute: bool) -> float:
    """Evaluate the log-likelihood for a given parameter combination.
    
    Main function that orchestrates the likelihood calculation: loads data,
    computes statistics, generates PDFs, computes UVLF, and evaluates the
    likelihood comparing model predictions with observations.

    Parameters
    ----------
    i : int
        Parameter index (for identification/debugging).
    data_directory : str
        Directory containing the Galacticus output files for this parameter
        combination.
    skewed : bool
        If True, uses skewed (two-sided normal) distributions; if False, uses
        standard normal distributions.
    reload : bool
        If True, forces reload of data from HDF5 files.
    recompute : bool
        If True, forces recomputation of PDFs and UVLF even if cached files
        exist.

    Returns
    -------
    float
        Log-likelihood value. Returns np.nan if an error occurs during
        calculation (e.g., data not available).

    Notes
    -----
    - Computes both absolute and apparent UVLFs (absolute is for plotting)
    - Uses global variables: absolute_magnitude_grid, apparent_magnitude_grid,
      binned_weights, dabs
    """
    # try:
    data = load_data(data_directory, reload)
    uvlf_filename = path.join(data_directory, 'apparent_uvlf.npy')
    if not recompute and path.isfile(uvlf_filename):
        abs_probs = None 
        app_probs = None
    else:
        if skewed:
            stats = get_skewed_stats(data)        
            abs_probs = get_skewed_probs(absolute_magnitude_grid, stats, data_directory, True, recompute)
            app_probs = get_skewed_probs(apparent_magnitude_grid, stats, data_directory, False, recompute)
        else:
            stats = get_stats(data)        
            abs_probs = get_probs(absolute_magnitude_grid, stats, data_directory, True, recompute)
            app_probs = get_probs(apparent_magnitude_grid, stats, data_directory, False, recompute)

    # Absolute UVLF is computed for plotting purposes (not used in likelihood)
    abs_uvlf = get_uvlf(abs_probs, binned_weights, data_directory, True, skewed, recompute)/dabs
    app_uvlf = get_uvlf(app_probs, binned_weights, data_directory, False, skewed, recompute)
    loglike = calculate_likelihood(app_uvlf)
    # except Exception as e:
    #     print(i)
    #     print(e)
    #     loglike = np.nan
    return loglike


def get_astro_params(dirname: str, initial: int, final: int) -> tuple:
    """Extract astrophysical parameters from Galacticus XML files.
    
    Reads parameter values from the XML input files used for Galacticus
    simulations. Extracts four key parameters: outflow velocity, outflow
    alpha, star formation timescale, and star formation alpha.

    Parameters
    ----------
    dirname : str
        Base directory name pattern (e.g., 'paper_params').
    initial : int
        Starting parameter index.
    final : int
        Ending parameter index (exclusive).

    Returns
    -------
    tuple
        Tuple of four lists:
        - outflow_velocities: list of outflow velocity parameters
        - outflow_alphas: list of outflow alpha parameters
        - sfr_alphas: list of star formation alpha parameters
        - sfr_timescales: list of star formation timescale parameters

    Notes
    -----
    - Hardcoded base path: '/carnegie/nobackup/users/gdriskell/jwst_data/'
    - Reads from z8.0.xml file in each parameter directory
    - XML paths are hardcoded for specific Galacticus parameter locations
    """
    base = '/carnegie/nobackup/users/gdriskell/jwst_data/'
    Vout_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic'
    alphaOut_xml = 'nodeOperator/nodeOperator/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent'
    tau0_xml = 'starFormationRateDisks/starFormationTimescale/timescale'
    alphaStar_xml = 'starFormationRateDisks/starFormationTimescale/exponentVelocity'

    outflow_velocities = []
    outflow_alphas = []
    sfr_alphas = []
    sfr_timescales = []

    for i in range(initial, final):
        data_directory = path.join(base, dirname+f'_p{i}/')
        xml_filename = path.join(data_directory,'z8.0.xml')
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        outflow_velocities.append(float(root.find(Vout_xml).get('value')))
        outflow_alphas.append(float(root.find(alphaOut_xml).get('value')))
        sfr_timescales.append(float(root.find(tau0_xml).get('value')))
        sfr_alphas.append(float(root.find(alphaStar_xml).get('value')))
        
    return outflow_velocities, outflow_alphas, sfr_alphas, sfr_timescales


def run(dirname: str, base: str, i: int, skewed: bool, 
        reload: bool, recompute: bool) -> float:
    """Wrapper function to evaluate likelihood for a single parameter combination.
    
    Convenience function that constructs the data directory path and calls
    evaluate_likelihood. Used for parallel processing with joblib.

    Parameters
    ----------
    dirname : str
        Base directory name pattern (e.g., 'paper_params').
    base : str
        Base path to data directories.
    i : int
        Parameter index.
    skewed : bool
        Whether to use skewed distributions.
    reload : bool
        Whether to force reload of data.
    recompute : bool
        Whether to force recomputation of PDFs/UVLF.

    Returns
    -------
    float
        Log-likelihood value.
    """
    data_directory = path.join(base, dirname+f'_p{i}/')
    loglike = evaluate_likelihood(i, data_directory, skewed, reload, recompute)
    return loglike


def save_results(loglikes: list, initial: int, final: int, 
                 dirname: str, outfilename: str) -> None:
    """Save likelihood results to CSV file.
    
    Combines log-likelihood values with their corresponding astrophysical
    parameters and saves to a CSV file. Results are sorted by log-likelihood
    (descending) and include both log-likelihood and likelihood (exp of log).

    Parameters
    ----------
    loglikes : list
        List of log-likelihood values for each parameter combination.
    initial : int
        Starting parameter index.
    final : int
        Ending parameter index (inclusive).
    dirname : str
        Base directory name pattern (e.g., 'paper_params').
    outfilename : str
        Output filename (without .csv extension). If empty, uses dirname.

    Returns
    -------
    None
        Saves results to CSV file and prints top 10 results to console.
    """
    (outflow_velocities, outflow_alphas, sfr_alphas, 
            sfr_timescales) = get_astro_params(dirname, initial, final+1)
    idxs = list(range(initial, final+1))
    columns = ['outflow_velocity', 'outflow_alpha', 'sfr_timescale', 
                'sfr_alpha', 'loglike']
    data = np.array([outflow_velocities, outflow_alphas, sfr_timescales, 
                        sfr_alphas, loglikes]).T
    df = pd.DataFrame(data, columns=columns, index=idxs)
    if len(outfilename)>0:
        output_csv_filename = f'{outfilename}.csv'
    else:
        output_csv_filename = f'{dirname}.csv'
    print(f'Saving to {output_csv_filename}')

    df.rename(columns={'Unnamed: 0':'idx'},inplace=True)
    df.insert(len(df.columns), 'like', np.exp(df['loglike']))
    df = df.sort_values('loglike', ascending=False)

    df.to_csv(output_csv_filename)    
    print(df.head(n=10))


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze JWST simulated data from Galacticus")
    parser.add_argument("dirname", help="Path to data directory")
    parser.add_argument("--base", type=str,  help="Base directory for data files")
    parser.add_argument("--outfilename", type=str, default='', help="Output filename")
    parser.add_argument("--initial", type=int, help="Initial index to run")
    parser.add_argument("--final", type=int, help="Final index to run")
    parser.add_argument("--save", action='store_true', help="Whether to save output to file")
    parser.add_argument("--skewed", action='store_true', help="Whether to use skewed pdf for analysis")
    parser.add_argument("--reload", action='store_true', help="Whether to reload data")
    parser.add_argument("--recompute", action='store_true', help="Whether to recompute probs and uvlf")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    args = parser.parse_args()

    start = time()
    loglikes = Parallel(n_jobs=args.n_jobs)(delayed(run)(args.dirname, args.base, i, args.skewed,
         args.reload, args.recompute,) for i in range(args.initial, args.final+1))
    print(f'for n={args.final-args.initial+1} total time = {time()-start}')

    if args.save:
       save_results(loglikes, args.initial, args.final, args.dirname, 
                    args.outfilename)

