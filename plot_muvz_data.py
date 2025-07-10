import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import analysis
import os.path as path

mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['figure.titlesize'] = 23
mpl.rcParams['xtick.major.size'] = 8#10
mpl.rcParams['ytick.major.size'] = 8#10
mpl.rcParams['xtick.minor.size'] = 6
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['font.family'] = 'DeJavu Serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'


def get_ci(pdf,frac):
    flat_zz = zz.flatten()
    flat_mm = mm.flatten()
    flat_pdf = pdf.flatten()
    # area = 0
    sorted_idx = np.argsort(flat_pdf)[::-1]
    sorted_pdf = np.sort(flat_pdf)[::-1]
    pdf_cumsum = np.cumsum(sorted_pdf)
    # print(sorted_pdf[:5])
    # print(pdf_cumsum[:5])
    result = []

    cidx = np.argmin(np.abs(pdf_cumsum-frac))
    # print(cidx)
    ci1_z = flat_zz[sorted_idx][:cidx+1]
    ci1_m = flat_mm[sorted_idx][:cidx+1]

    contour_maxes = []
    contour_zs = []
    contour_mins = []
    for z in analysis.redshift_grid:
        zi = ci1_z == z
        if np.sum(zi) > 0:
            m_max = np.amax(ci1_m[zi])
            m_min = np.amin(ci1_m[zi])
            contour_maxes.append(m_max)
            contour_mins.append(m_min)
            contour_zs.append(z)
    return contour_maxes, contour_mins, contour_zs


def calculate_survey_pdf(completeness, pdf, pmh, out_fn):
    if path.isfile(out_fn):
        return np.load(out_fn)
    pdf = np.sum(pmh * pdf, axis=2)
    pdf = completeness * pdf 
    pdf /= pdf.sum()
    np.save(out_fn, pdf)
    return pdf


f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True)

ngdeep_data = pd.read_csv('data/ngdeep_data.csv')
ngdeep_data = ngdeep_data[ngdeep_data['mf277w'] < 30.4]
ngdeep_data.reset_index(inplace=True)
ceers_data = pd.read_csv('data/CEERS_data.csv')
ceers_data = ceers_data[ceers_data['mf277w'] < 29.15]
ceers_data.reset_index(inplace=True)
n_ceers = len(ceers_data)
n_ngdeep = len(ngdeep_data)

ceers_data['z_regrid'] = np.round(ceers_data['z']*2)/2
ngdeep_data['z_regrid'] = np.round(ngdeep_data['z']*2)/2

mm,zz = np.meshgrid(analysis.apparent_magnitude_grid,analysis.redshift_grid[1:],indexing='ij')

p1 = np.e**(-0.5)
p2 = np.e**(-2.)
p3 = np.e**(-9./2.)
p4 = np.e**(-16./2.)
p5 = np.e**(-25./2.)

pmh = analysis.binned_weights / analysis.binned_weights.sum(axis=-1).reshape(-1,1)

base_dir = '/carnegie/scidata/groups/dmtheory/jwst_simulated_data/paper_params_p13845/'
pdf_path = path.join(base_dir, 'apparent_pdf.npy')
pdf = np.load(pdf_path)

out_fn = path.join(base_dir, 'ceers_pdf.npy')
ceers_theory_pdf  = calculate_survey_pdf(analysis.ceers_effective_volume, pdf, pmh, out_fn)

out_fn = path.join(base_dir, 'ngdeep_pdf.npy')
ngdeep_theory_pdf = calculate_survey_pdf(analysis.ngdeep_effective_volume, pdf, pmh, out_fn)

levels = [p1,p2,p3][::-1] 

idx = analysis.redshift_grid > 8.4
ceers_theory_pdf = ceers_theory_pdf[:,idx]
ngdeep_theory_pdf = ngdeep_theory_pdf[:,idx]
# ceers_theory_pdf /= ceers_theory_pdf.max()
ceers_theory_pdf /= ceers_theory_pdf.sum()
# ngdeep_theory_pdf /= ngdeep_theory_pdf.max()
ngdeep_theory_pdf /= ngdeep_theory_pdf.sum()

ceers_color_norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(ceers_theory_pdf))
ngdeep_color_norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(ngdeep_theory_pdf))

cmap = 'Reds'

axs[0].pcolormesh(zz, mm, ngdeep_theory_pdf, cmap = 'Reds', norm = ngdeep_color_norm, 
    shading = 'gouraud', label='NGDEEP Theory prediction')
axs[1].pcolormesh(zz, mm, ceers_theory_pdf, cmap = 'Blues', norm = ceers_color_norm, 
    shading = 'gouraud', label='CEERS Theory prediction')

axs[0].errorbar(ngdeep_data['z'], ngdeep_data['mf277w'], xerr=[-ngdeep_data['z_lower_err'], ngdeep_data['z_upper_err']], 
                color="k", ecolor='k', ls='none', capsize=3.5, marker='o', label='NGDEEP Obs.') # color='tab:red',

annotation_cmap = 'binary'
norm = mpl.colors.Normalize(vmin=0, vmax=1) 
sm = mpl.cm.ScalarMappable(cmap=annotation_cmap, norm=norm)

for frac in [0.68, 0.95, 0.99]:
    if np.isclose(frac,0.68):
        c = sm.to_rgba(0.80)
    elif np.isclose(frac,0.95):
        c = sm.to_rgba(0.50)
    else:
        c = sm.to_rgba(0.30)
    maxes,mins,zs = get_ci(ngdeep_theory_pdf,frac)
    axs[0].plot(zs, maxes, '-', color=c, zorder=999)
    axs[0].plot(zs, mins, '-', color=c, zorder=999)
    axs[0].plot([zs[-1],zs[-1]], [mins[-1],maxes[-1]], '-', color=c, zorder=999)
    total = 0
    for i,z in enumerate(zs):
        idx = (ngdeep_data['z_regrid'] == z) & (ngdeep_data['mf277w'] < maxes[i]) & (ngdeep_data['mf277w'] > mins[i])
        total += np.sum(idx)
    print(frac,total/n_ngdeep)
        

axs[1].errorbar(ceers_data['z'], ceers_data['mf277w'], xerr=[-ceers_data['z_lower_err'], ceers_data['z_upper_err']],
                color="k", ecolor='k', ls='none', capsize=3.5, marker='o', label='CEERS Obs.') # color='tab:red',

for frac in [0.68, 0.95, 0.99]:
    maxes,mins,zs = get_ci(ceers_theory_pdf,frac)
    if np.isclose(frac,0.68):
        c = sm.to_rgba(0.8)
    elif np.isclose(frac,0.95):
        c = sm.to_rgba(0.5)
    else:
        c = sm.to_rgba(0.3)
    # print(maxes, mins, zs)
    axs[1].plot(zs, maxes, '-', color=c, zorder=999)
    axs[1].plot(zs, mins, '-', color=c, zorder=999)
    axs[1].plot([zs[-1],zs[-1]], [mins[-1],maxes[-1]], '-', color=c, zorder=999)
    total = 0
    for i,z in enumerate(zs):
        idx = (ceers_data['z_regrid'] == z) & (ceers_data['mf277w'] < maxes[i]) & (ceers_data['mf277w'] > mins[i])
        total += np.sum(idx)
    print(frac,total/n_ceers)


axs[0].annotate(r'$1\sigma$', (9.6, 28.25), color=sm.to_rgba(0.8), fontsize=15, zorder=999)
axs[0].annotate(r'$2\sigma$', (9.6, 27.05), color=sm.to_rgba(0.5), fontsize=15, zorder=999)
axs[0].annotate(r'$3\sigma$', (9.6, 26.05), color=sm.to_rgba(0.3), fontsize=15, zorder=999)

axs[1].annotate(r'$1\sigma$', (9.25, 27.175), color=sm.to_rgba(0.8), fontsize=14.5, zorder=999)
axs[1].annotate(r'$2\sigma$', (9.25, 25.75), color=sm.to_rgba(0.5), fontsize=15, zorder=999)
axs[1].annotate(r'$3\sigma$', (9.25, 24.9), color=sm.to_rgba(0.3), fontsize=15, zorder=999)

axs[0].set_xlabel(r'$z$')
axs[1].set_xlabel(r'$z$')
axs[0].set_ylabel(r'$m_{\mathrm{UV}}$')
axs[1].set_ylabel(r'$m_{\mathrm{UV}}$')
axs[0].set_xlim(8.5, 16)
axs[1].set_xlim(8.5, 16)
axs[0].set_ylim(24.5, 30.5)
axs[1].set_ylim(24.5, 30.5)
axs[0].legend(frameon=False, fontsize=12)
axs[1].legend(frameon=False, fontsize=12)
plt.savefig('muvz_viz.pdf') 
