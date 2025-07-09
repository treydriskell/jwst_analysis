import numpy as np
# import ndtest
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import analysis

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

f, axs = plt.subplots(1, 2, figsize=(12,5),constrained_layout=True)


#ngdeep_pdf = np.load('/home/gdriskell/galacticus/data/ngdeep_pdf.npy')
#ceers_pdf = np.load('/home/gdriskell/galacticus/data/ceers_pdf.npy')
#data_pdf = ngdeep_pdf+ceers_pdf
#data_pdf = np.sum(data_pdf, axis=0) / len(data_pdf)

ngdeep_data = pd.read_csv('data/ngdeep_data.csv')
ngdeep_data = ngdeep_data[ngdeep_data['mf277w'] < 30.4]
ngdeep_data.reset_index(inplace=True)
ceers_data = pd.read_csv('data/CEERS_data.csv')
ceers_data = ceers_data[ceers_data['mf277w'] < 29.15]
ceers_data.reset_index(inplace=True)
# print(len(ceers_data),len(ngdeep_data))
n_ceers = len(ceers_data)
n_ngdeep = len(ngdeep_data)

ceers_data['z_regrid'] = np.round(ceers_data['z']*2)/2
ngdeep_data['z_regrid'] = np.round(ngdeep_data['z']*2)/2

# n_ceers

# data = pd.concat([ngdeep_data, ceers_data])
# obs_muv = np.concatenate((ngdeep_data['mf277w'],ceers_data['mf277w']))
# obs_z = np.concatenate((ngdeep_data['z'],ceers_data['z']))
# obs_z = np.concatenate((ngdeep_data['z'],ceers_data['z']))
# print(len(obs_muv),len(obs_z))


# print()
# print(theory_pdf[:5,0])
# print(theory_pdf.flatten()[0:5])
# nz = 17
# z_grid = np.linspace(8.0, 16.0, nz)
# dz = z_grid[1]-z_grid[0]

# app_min = 22.0
# app_max = 45.0
# dapp = 0.25 #0.2
# napp = int(round((app_max-app_min)/dapp))+1
# app_grid = np.linspace(app_min, app_max, napp) 
# dapp = app_grid[1]-app_grid[0]

mm,zz = np.meshgrid(analysis.apparent_magnitude_grid,analysis.redshift_grid[1:],indexing='ij')
# ceers_grid = np.stack((mm,zz,ceers_theory_pdf),axis=-1).reshape(-1,3)
# ngdeep_grid = np.stack((mm,zz,ngdeep_theory_pdf),axis=-1).reshape(-1,3)
p1 = np.e**(-0.5)
p2 = np.e**(-2.)
p3 = np.e**(-9./2.)
p4 = np.e**(-16./2.)
p5 = np.e**(-25./2.)

ceers_theory_pdf = np.load('/carnegie/scidata/groups/dmtheory/jwst_simulated_data/paper_params_p13845/ceers_pdf.npy')#.reshape(napp,nz)
ngdeep_theory_pdf = np.load('/carnegie/scidata/groups/dmtheory/jwst_simulated_data/paper_params_p13845/ngdeep_pdf.npy')#.reshape(napp,nz)

levels = [p1,p2,p3][::-1] #p4,p5

# ceers_theory_pdf[ceers_theory_pdf>=p1] = 1.0
# ceers_theory_pdf[(ceers_theory_pdf<p1)  & (ceers_theory_pdf>=p2)] = 0.75
# ceers_theory_pdf[(ceers_theory_pdf<p2)  & (ceers_theory_pdf>=p3)] = 0.50
# ceers_theory_pdf[(ceers_theory_pdf<p3)  & (ceers_theory_pdf>=p4)] = 0.25
# ceers_theory_pdf[(ceers_theory_pdf<p3)  & (ceers_theory_pdf>=p5)] = 0.1


idx = analysis.redshift_grid > 8.4
ceers_theory_pdf = ceers_theory_pdf[:,idx]
ngdeep_theory_pdf = ngdeep_theory_pdf[:,idx]
# ceers_theory_pdf /= ceers_theory_pdf.max()
ceers_theory_pdf /= ceers_theory_pdf.sum()
# ngdeep_theory_pdf /= ngdeep_theory_pdf.max()
ngdeep_theory_pdf /= ngdeep_theory_pdf.sum()

ceers_color_norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(ceers_theory_pdf))
# ceers_sm = ceers_color_norm.
ngdeep_color_norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(ngdeep_theory_pdf))

# plt.imshow(mm,zz,)
# plt.imshow
# cmap = cmr.get_sub_cmap('cmr.neutral_r', 0.05, 1.0)
cmap = 'Reds'

# print(zz.min())

# print(zz.shape, mm.shape, ngdeep_theory_pdf.shape)
axs[0].pcolormesh(zz, mm, ngdeep_theory_pdf, cmap = 'Reds', norm = ngdeep_color_norm, 
    shading = 'gouraud', label='NGDEEP Theory prediction')
axs[1].pcolormesh(zz, mm, ceers_theory_pdf, cmap = 'Blues', norm = ceers_color_norm, 
    shading = 'gouraud', label='CEERS Theory prediction')

# axs[0].scatter(ngdeep_data['z'], ngdeep_data['mf277w'], marker='x', color='gray', label='NGDEEP Obs.') # color='tab:red',
axs[0].errorbar(ngdeep_data['z'], ngdeep_data['mf277w'], xerr=[-ngdeep_data['z_lower_err'], ngdeep_data['z_upper_err']], 
                color="k", ecolor='k', ls='none', capsize=3.5, marker='o', label='NGDEEP Obs.') # color='tab:red',
# cs0 = axs[0].contour(zz, mm, ngdeep_theory_pdf, levels=levels, colors='gray')

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
    # print(maxes, mins, zs)
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


# cs1 = axs[1].contour(zz, mm, ceers_theory_pdf, levels=levels, colors='gray')
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
# print()
plt.savefig('muvz_viz.pdf') 

# print(sorted_idx[0])
# while area < 0.68:
    # area += flat_pdf[]




# # ngdeep
# for i in range(len(levels)-1):
#     print(f'level prob {levels[i]}')
#     cs = axs[0].contour(zz, mm, ngdeep_theory_pdf, levels=[levels[i]], colors='gray')
    
#     p = cs.collections[0].get_paths()[0]
#     v = p.vertices
#     x = v[:,0]
#     y = v[:,1]

#     # print(x_left, y_left)
#     # raise Exception()

#     # cs_right = axs[0].contour(zz, mm, ngdeep_theory_pdf, levels=[levels[i+1]], colors='gray')
#     # p_right = cs_right.collections[0].get_paths()[0]
#     # v_right = p_left.vertices
#     # x_right = v_left[:,0]
#     # y_right = v_left[:,1]

#     z_max = np.amax(x)
#     z_max_idx = np.argmax(x)
#     lower_mags = y[:z_max_idx]
#     lower_zs = x[:z_max_idx]
#     upper_mags = y[z_max_idx:]
#     upper_zs = x[z_max_idx:]
#     # print(lower_mags,upper_mags)

#     data_idx = ngdeep_data['z']<z_max
#     data_zs = ngdeep_data['z'][data_idx]
#     data_muvs = ngdeep_data['mf277w'][data_idx]

#     # sort_idx = 

#     # print(data_zs)

#     lower_interp = np.interp(data_zs, lower_zs, lower_mags)
#     upper_interp = np.interp(data_zs, upper_zs[::-1], upper_mags[::-1])
#     # print(lower_interp)
#     # print(upper_interp)
#     n_1s = np.sum((data_muvs > lower_interp) & (data_muvs < upper_interp))/len(ngdeep_data['z'])
#     print(n_1s)
