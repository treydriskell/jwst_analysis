# Working with the raw data:
The data is currently hosted on OBS at the directory `/carnegie/nobackup/users/gdriskell/jwst_data/`. The raw data and data products for each parameter combination is saved to a seperate subdirectory labeled `.../jwst_data/paper_params_p{i}` where i runs from 0 to 73599. The best-fit parameter from our analysis for example is 13845 so is located in subdirectory `paper_params_p13845`. The relevant outputs from galacticus are saved as `paper_params_p{i}/z{z}.xml` (for the input parameter file) and `paper_params_p{i}/z{z}.hdf5` (for the data), where z $\in$ ["8.0", "12.0", "16.0"], e.g. `paper_params_p0/z8.0.xml`. The analysis script will process the data from the hdf5 file.

analysis.py loops through files and performs all the necessary analysis, saving results to <outfilename>.csv. Use "python analysis.py --help" to see which arguments need to be passed running analysis. 

Plots for the figure are generated as follows:
- Figure 2,4,6,8: run plotting.py
- Figure 3: run astro_uvlf.py
- Figure 5: run plot_muvz_data.py
- Figure 7: run galform_comparison.py
- Figure 9: run plot_hst_uvlf.py (left) and smf_comp.py (right)
