
import h5py
import numpy as np
# import os
import os.path as path
from astropy.cosmology import FlatLambdaCDM
import itertools
import analysis

rng = np.random.default_rng()

# global cosmo
cosmo = FlatLambdaCDM(H0=70.000, Om0=0.286, Tcmb0=2.72548, Ob0=0.047)

z9_upper = np.zeros_like(analysis.absolute_magnitude_grid)
z9_lower = np.ones_like(analysis.absolute_magnitude_grid)
z11_upper = np.zeros_like(analysis.absolute_magnitude_grid)
z11_lower = np.ones_like(analysis.absolute_magnitude_grid)

base_dir = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data'

bestfit_directory = path.join(base_dir, 'paper_params_p13845/')
bestfit_data = analysis.load_data(bestfit_directory, True, False)
bestfit_stats = analysis.get_stats(bestfit_data)
bestfit_abs_probs = analysis.get_probs(analysis.absolute_magnitude_grid, bestfit_stats, 
                                    bestfit_directory, True, False, False)
uvlf = analysis.get_uvlf(bestfit_abs_probs, analysis.binned_weights, bestfit_directory, True, False, False)
uvlf /= analysis.dabs

app_cutoff = 30.4

z_left = analysis.redshift_grid - analysis.dz/2.0
z_right = analysis.redshift_grid + analysis.dz/2.0

zidx = (z_left >= 8.5) & (z_right <= 9.5)
zs = analysis.redshift_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = analysis.z_volumes[zidx]
totalV = np.zeros_like(analysis.absolute_magnitude_grid)
y = np.zeros_like(analysis.absolute_magnitude_grid)
for j,muv in enumerate(analysis.absolute_magnitude_grid):
    for k in range(len(zs)):
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
y /= totalV
bestfit_z9 = y

zidx = (z_left > 9.5) & (z_right < 12.0)
zs = analysis.redshift_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = analysis.z_volumes[zidx]
totalV = np.zeros_like(analysis.absolute_magnitude_grid)
y = np.zeros_like(analysis.absolute_magnitude_grid)
for j,muv in enumerate(analysis.absolute_magnitude_grid):
    for k in range(len(zs)):
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
y /= totalV
bestfit_z11 = y

zidx = (z_left > 13) & (z_right <= 15.0)
zs = analysis.redshift_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = analysis.z_volumes[zidx]
totalV = np.zeros_like(analysis.absolute_magnitude_grid)
y = np.zeros_like(analysis.absolute_magnitude_grid)
for j,muv in enumerate(analysis.absolute_magnitude_grid):
    for k in range(len(zs)):
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
y /= totalV
bestfit_z14 = y


ngdeep_muv = [-20.1, -19.1, -18.35, -17.85, -17.35]
ngdeep_phi = np.array([14.7e-5, 18.9e-5, 74.0e-5, 170.0e-5, 519.0e-5])
ngdeep_phi_err_lower = np.array([7.2e-5, 8.9e-5, 29.0e-5, 65.e-5, 198.e-5])
ngdeep_phi_err_upper = np.array([11.1e-5, 13.8e-5, 41.4e-5, 85.e-5, 248.e-5])

ceers_muv_z9 = [-22.0, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5]
ceers_phi_z9 = np.array([1.1e-5, 2.2e-5, 8.2e-5, 9.6e-5, 28.6e-5, 26.8e-5, 136.0e-5])
ceers_upper_z9 = np.array([0.7e-5, 1.3e-5, 4.0e-5, 4.6e-5, 11.5e-5, 12.4e-5, 61.0e-5])
ceers_lower_z9 = np.array([0.6e-5, 1.0e-5, 3.2e-5, 3.6e-5, 9.1e-5, 10.0e-5,49.9e-5])

ngdeep_chi2_z9 = np.zeros_like(ngdeep_phi)
# chi2_z9 = 0

for i,muv in enumerate(ngdeep_muv):
    midx = np.argmin(np.abs(analysis.absolute_magnitude_grid-muv))
    bf_i = bestfit_z9[midx]
    ng_i = ngdeep_phi[i]
    if bf_i>ng_i:
        ngdeep_chi2_z9[i] = (bf_i-ng_i)**2 / ngdeep_phi_err_upper[i]**2
    else:
        ngdeep_chi2_z9[i] = (bf_i-ng_i)**2 / ngdeep_phi_err_lower[i]**2

ng_z9 = len(ngdeep_chi2_z9)
print(ngdeep_chi2_z9)
reduced_ngdeep_chi2_z9 = np.sum(ngdeep_chi2_z9)/ng_z9
print(f'NGDeep z=9 chi2 {reduced_ngdeep_chi2_z9}')
chi2_z9 = reduced_ngdeep_chi2_z9*ng_z9

ceers_chi2_z9 = np.zeros_like(ceers_phi_z9)

for i,muv in enumerate(ceers_muv_z9):
    midx = np.argmin(np.abs(analysis.absolute_magnitude_grid-muv))
    bf_i = bestfit_z9[midx]
    c_i = ceers_phi_z9[i]
    if bf_i>c_i:
        ceers_chi2_z9[i] = (bf_i-c_i)**2 / ceers_upper_z9[i]**2
    else:
        ceers_chi2_z9[i] = (bf_i-c_i)**2 / ceers_lower_z9[i]**2

print(ceers_chi2_z9)
nc_z9 = len(ceers_chi2_z9)
reduced_ceers_chi2_z9 = np.sum(ceers_chi2_z9)/(nc_z9)
print(f'Ceers z=9 chi2 {reduced_ceers_chi2_z9}')
chi2_z9 += reduced_ceers_chi2_z9*nc_z9
print(f'Total z=9 chi2 = {chi2_z9/(nc_z9+ng_z9-4)}')

ngdeep_muv = [-19.35, -18.65, -17.95, -17.25]
ngdeep_phi = np.array([18.5e-5, 27.7e-5, 59.1e-5, 269.0e-5])
ngdeep_phi_err_lower = np.array([8.3e-5, 13.0e-5, 29.3e-5, 124.e-5])
ngdeep_phi_err_upper = np.array([11.9e-5, 18.3e-5, 41.9e-5, 166.e-5])

ceers_muv_z11 = [-20.5, -20.0, -19.5, -19.0, -18.5]
ceers_phi_z11 = np.array([1.8e-5, 5.4e-5, 7.6e-5, 17.6e-5, 26.3e-5])
ceers_upper_z11 = np.array([1.2e-5, 2.7e-5, 3.9e-5, 10.3e-5, 18.2e-5])
ceers_lower_z11 = np.array([0.9e-5, 2.1e-5, 3.0e-5, 7.9e-5, 13.3e-5])

ceers_muv_z14 = [-20.0, -19.5]
ceers_phi_z14 = np.array([2.6e-5, 7.3e-5])
ceers_upper_z14 = np.array([3.3e-5, 6.9e-5])
ceers_lower_z14 = np.array([1.8e-5, 4.4e-5])


ngdeep_chi2_z11 = np.zeros_like(ngdeep_phi)

for i,muv in enumerate(ngdeep_muv):
    midx = np.argmin(np.abs(analysis.absolute_magnitude_grid-muv))
    bf_i = bestfit_z11[midx]
    ng_i = ngdeep_phi[i]
    if bf_i>ng_i:
        ngdeep_chi2_z11[i] = (bf_i-ng_i)**2 / ngdeep_phi_err_upper[i]**2
    else:
        ngdeep_chi2_z11[i] = (bf_i-ng_i)**2 / ngdeep_phi_err_lower[i]**2

print(ngdeep_chi2_z11)
ng_z11 = len(ngdeep_chi2_z11)
reduced_ngdeep_chi2_z11 = np.sum(ngdeep_chi2_z11)/ng_z11
print(f'NGDeep z=11 chi2 {reduced_ngdeep_chi2_z11}')
chi2_z11 = reduced_ngdeep_chi2_z11*ng_z11

ceers_chi2_z11 = np.zeros_like(ceers_phi_z11)

for i,muv in enumerate(ceers_muv_z11):
    midx = np.argmin(np.abs(analysis.absolute_magnitude_grid-muv))
    bf_i = bestfit_z11[midx]
    c_i = ceers_phi_z11[i]
    if bf_i>c_i:
        ceers_chi2_z11[i] = (bf_i-c_i)**2 / ceers_upper_z11[i]**2
    else:
        ceers_chi2_z11[i] = (bf_i-c_i)**2 / ceers_lower_z11[i]**2

print(ceers_chi2_z11)
nc_z11 = len(ceers_chi2_z11)
reduced_ceers_chi2_z11 = np.sum(ceers_chi2_z11)/(nc_z11)
print(f'Ceers z=11 chi2 {reduced_ceers_chi2_z11}')
chi2_z11 += reduced_ceers_chi2_z11*nc_z11
print(f'Total z=11 chi2 = {chi2_z11/(nc_z11+ng_z11-4)}')

ceers_chi2_z14 = np.zeros_like(ceers_phi_z14)

for i,muv in enumerate(ceers_muv_z14):
    midx = np.argmin(np.abs(analysis.absolute_magnitude_grid-muv))
    bf_i = bestfit_z14[midx]
    c_i = ceers_phi_z14[i]
    if bf_i>c_i:
        ceers_chi2_z14[i] = (bf_i-c_i)**2 / ceers_upper_z14[i]**2
    else:
        ceers_chi2_z14[i] = (bf_i-c_i)**2 / ceers_lower_z14[i]**2
chi2_z14 = np.sum(ceers_chi2_z14)
nc_z14 = len(ceers_chi2_z14)

print(f'Total chi2 = {(chi2_z11+chi2_z9+chi2_z14)/(nc_z11+ng_z11+nc_z9+ng_z9+nc_z14-4)}')

# print(nc_z11+ng_z11+nc_z9+ng_z9+nc_z14)


# HST chi2

Bouwens2021_Muv = [-21.85, -21.35, -20.85, -20.10, -19.35, -18.60, -17.60]
Bouwens2021_phi = np.array([0.000003, 0.000012, 0.000041, 0.000120, 0.000657, 0.001100, 0.003020])
Bouwens2021_err = np.array([0.000002, 0.000004, 0.000011, 0.000040, 0.000233, 0.000340, 0.001140])
Bouwens2021_log_phi = np.log10(Bouwens2021_phi)

Bowler2020_Muv = [-21.65, -22.15, -22.90]
Bowler2020_phi = np.array([2.95e-6, 0.58e-6, 0.14e-6])
Bowler2020_err = np.array([0.98e-6, 0.33e-6, 0.06e-6])
Bowler2020_log_phi = np.log10(Bowler2020_phi)

Stefanon2019_Muv = [-22.55, -22.05, -21.55]
Stefanon2019_phi = np.array([0.76e-6, 1.38e-6, 4.87e-6])
Stefanon2019_err_upper = np.array([0.74e-6, 1.09e-6, 2.01e-6])
Stefanon2019_err_lower = np.array([0.41e-6, 0.66e-6, 1.41e-6])
Stefanon2019_log_phi = np.log10(Stefanon2019_phi)


Mclure2013_Muv = [-21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25, -17.75, -17.25]
Mclure2013_phi = np.array([0.000008, 0.00003, 0.0001, 0.0003, 0.0005, 0.0012, 0.0018, 0.0028, 0.0050])
Mclure2013_err = np.array([0.000003, 0.000009, 0.00003, 0.00006, 0.00012, 0.0004, 0.0006, 0.0008, 0.0025])
Mclure2013_log_phi = np.log10(Mclure2013_phi)


zidx = analysis.redshift_grid == 8.0
zs = analysis.redshift_grid[zidx]
z_left_slice = z_left[zidx]
z_right_slice = z_right[zidx]
zuvlf = uvlf[:,zidx]
zvolume = analysis.z_volumes[zidx]
totalV = np.zeros_like(analysis.absolute_magnitude_grid)
y = np.zeros_like(analysis.absolute_magnitude_grid)
for j,muv in enumerate(analysis.absolute_magnitude_grid):
    for k in range(len(zs)):
        # left > right
        # left_cutoff = app_cutoff - cosmo.distmod(z_left_slice[j]).value + 2.5*np.log10(1+z_left_slice[j])
        right_cutoff = app_cutoff - cosmo.distmod(z_right_slice[k]).value + 2.5*np.log10(1+z_right_slice[k])
        if muv < right_cutoff:
            y[j] += zuvlf[j,k]*zvolume[k]
            totalV[j] += zvolume[k]
        # elif muv < left_cutoff:
        #     volume = 
y /= totalV
bestfit_z8 = y

muvs = [Bouwens2021_Muv, Bowler2020_Muv, Stefanon2019_Muv, Mclure2013_Muv]
phis = [Bouwens2021_phi, Bowler2020_phi, Stefanon2019_phi, Mclure2013_phi]
err_upper = [Bouwens2021_err, Bowler2020_err, Stefanon2019_err_upper, Mclure2013_err]
err_lower = [Bouwens2021_err, Bowler2020_err, Stefanon2019_err_lower, Mclure2013_err]


# hst_chi2 = np.zeros_like(ceers_phi_z14)
hst_chi2 = 0
n_hst = 0

for muv,phi,err_upper,err_lower in zip(muvs, phis, err_upper, err_lower):
    for i,m in enumerate(muv):
        # print(m)
        n_hst += 1
        midx = np.argmin(np.abs(analysis.absolute_magnitude_grid-m))
        bf_i = bestfit_z8[midx]
        hst_i = phi[i]
        # print(m, bf_i, hst_i,  err_upper[i],  err_lower[i])
        if bf_i>ng_i:
            hst_chi2 += (bf_i-hst_i)**2 / err_upper[i]**2
        else:
            hst_chi2 += (bf_i-hst_i)**2 / err_lower[i]**2
        # print(hst_chi2/n_hst)

print(hst_chi2,n_hst)
print(f'HST chi2 is {hst_chi2/n_hst}')

print("WE KNOW THAT JWST AND HST DATA ARE IN TENSION AT BRIGHT END, LARGE CHI2 MAY \
BE REASONABLE. USE AT YOUR OWN RISK")