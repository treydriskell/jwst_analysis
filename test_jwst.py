import numpy as np
import ndtest
import pandas as pd
import os
import analysis

ngdeep_data = pd.read_csv('data/ngdeep_data.csv')
ngdeep_data = ngdeep_data[ngdeep_data['mf277w'] < 30.4]
ngdeep_data.reset_index(inplace=True)
ceers_data = pd.read_csv('data/CEERS_data.csv')
ceers_data = ceers_data[ceers_data['mf277w'] < 29.15]
ceers_data.reset_index(inplace=True)
n_ceers = len(ceers_data)
n_ngdeep = len(ngdeep_data)

obs_muv = np.concatenate((ngdeep_data['mf277w'],ceers_data['mf277w']))
obs_z = np.concatenate((np.round(ngdeep_data['z']*2)/2,np.round(ceers_data['z']*2)/2))


ceers_theory_pdf = np.load('/carnegie/scidata/groups/dmtheory/jwst_simulated_data/paper_params_p13845/ceers_pdf.npy')
ngdeep_theory_pdf = np.load('/carnegie/scidata/groups/dmtheory/jwst_simulated_data/paper_params_p13845/ngdeep_pdf.npy')

mm,zz = np.meshgrid(analysis.apparent_magnitude_grid, analysis.redshift_grid,indexing='ij')
ceers_grid = np.stack((mm,zz,ceers_theory_pdf),axis=-1).reshape(-1,3)
ngdeep_grid = np.stack((mm,zz,ngdeep_theory_pdf),axis=-1).reshape(-1,3)

ceers_df = pd.DataFrame(ceers_grid , columns=['muv','z','prob'])
idx = ceers_df['z'] < 8.5
ceers_df['prob'][idx] = 0
ngdeep_df = pd.DataFrame(ngdeep_grid , columns=['muv','z','prob'])
idx = ngdeep_df['z'] < 8.5
ngdeep_df['prob'][idx] = 0
n_sample = len(obs_muv)

ceers_theory_sample = ceers_df.sample(n_ceers*1000, replace=True, weights='prob')
ceers_muv_sample = ceers_theory_sample['muv'].to_numpy()
ceers_z_sample = ceers_theory_sample['z'].to_numpy()
ngdeep_theory_sample = ngdeep_df.sample(n_ngdeep*1000, replace=True, weights='prob')
ngdeep_muv_sample = ngdeep_theory_sample['muv'].to_numpy()
ngdeep_z_sample = ngdeep_theory_sample['z'].to_numpy()

theory_muv = np.concatenate((ceers_muv_sample,ngdeep_muv_sample))
theory_z = np.concatenate((ceers_z_sample,ngdeep_z_sample))

P_ceers = 0
P_ngdeep = 0
P_total = 0

P, D = ndtest.ks2d2s(ngdeep_data['mf277w'], np.round(ngdeep_data['z']*2)/2, ngdeep_muv_sample, ngdeep_z_sample, extra=True)
print(f"NGDEEP {P=:.3g}, {D=:.3g}")
P_ngdeep += P

P, D = ndtest.ks2d2s(ceers_data['mf277w'], np.round(ceers_data['z']*2)/2, ceers_muv_sample, ceers_z_sample, extra=True)
print(f"CEERS {P=:.3g}, {D=:.3g}")
P_ceers += P

P, D = ndtest.ks2d2s(obs_muv, obs_z, theory_muv, theory_z, extra=True)
print(f"{P=:.3g}, {D=:.3g}")
P_total += P

# print(f'Ngdeep P={P_ngdeep/1000.}, CEERS P={P_ceers/1000.}, Total P={P_total/1000.}')

